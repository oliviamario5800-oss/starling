//! Stage 2 spike, step 3a: an OpenXR **session** that clears the two eyes to
//! different solid colours — raw Vulkan via `ash`, no wgpu involvement.
//!
//! Step 2 (`xr_smoke`) proved the loader and instance handshake. This step
//! proves the rest of the chain from inside the game process: Vulkan device
//! negotiation, session creation, swapchains, the frame loop, and frame
//! submission to the compositor. Whether the compositor actually shows the
//! session in the headset (the manifest question from the design doc) is
//! answered by the person wearing it; the log markers (`XR_SESSION: …`,
//! tag `starling`) record how far the chain got either way.
//!
//! Step 3b will redo this clear through wgpu-wrapped swapchain textures
//! (the Helio pattern). Everything here is self-contained on one thread:
//! the instance, Vulkan device, session and swapchains are created, used
//! and destroyed on the `xr-session` thread, and the flat game is never
//! touched. Runs once per process, main process only (the `:launcher`
//! process shares `android::main` and must not hold a session), and can be
//! disabled with `HORIZON_NO_XR_SESSION`.
//!
//! Vulkan negotiation uses the legacy `XR_KHR_vulkan_enable` flow: the
//! `openxr` crate (0.22.0) wraps `create_vulkan_instance` for enable2 but
//! has no wrapper for `xrGetVulkanDeviceExtensions2KHR`, so the enable2
//! device-extension list is unreachable through its high-level API. The
//! legacy getters are all wrapped. See stage2-smoke/OPEN_QUESTIONS.md Q5.

use std::ffi::{c_char, c_void, CString};
use std::time::Instant;

use ash::vk::Handle as _;
use openxr::vulkan::Vulkan as XrVulkan;

/// Left eye: red. Right eye: green. Deliberately unmissable.
const EYE_COLOURS: [[f32; 4]; 2] = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]];
/// How long to wait for the runtime to offer a session before giving up.
const READY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);
/// Hard cap on the whole session once begun, so a forgotten headset does
/// not hold the thread (and a Vulkan device) forever.
const SESSION_CAP: std::time::Duration = std::time::Duration::from_secs(600);

fn log_info(message: &str) {
    crate::android::logcat(log::Level::Info, message);
}

fn log_warn(message: &str) {
    crate::android::logcat(log::Level::Warn, message);
}

/// Entry point, called from `android::main` next to the smoke probe.
/// Spawns the session thread; returns immediately.
pub fn start_once() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var_os("HORIZON_NO_XR_SESSION").is_some() {
            log_info("XR_SESSION: skipped (HORIZON_NO_XR_SESSION set)");
            return;
        }
        // Only the main process: `:launcher` shares this entry point.
        let cmdline = std::fs::read("/proc/self/cmdline").unwrap_or_default();
        if String::from_utf8_lossy(&cmdline).contains(':') {
            log_info("XR_SESSION: skipped (not the main process)");
            return;
        }
        let spawned = std::thread::Builder::new()
            .name("xr-session".to_owned())
            .spawn(run_entry);
        if let Err(error) = spawned {
            log_warn(&format!("XR_SESSION: unavailable: spawn: {error}"));
        }
    });
}

fn run_entry() {
    match run() {
        Ok(report) => log_info(&format!("XR_SESSION: ok {report}")),
        Err(error) => log_warn(&format!("XR_SESSION: unavailable: {error}")),
    }
}

type StageResult<T> = Result<T, String>;

fn stage<T>(stage: &str, result: Result<T, impl std::fmt::Display>) -> StageResult<T> {
    result.map_err(|error| format!("{stage}: {error}"))
}

fn run() -> StageResult<String> {
    // ---- OpenXR instance + HMD system (same pattern as xr_smoke) ----
    let app = crate::android::android_app();
    // SAFETY: the pointers are the JavaVM and activity of this process,
    // kept alive by the GameActivity for the process lifetime.
    let platform_info = unsafe {
        openxr::AndroidPlatformInfo::new(app.vm_as_ptr(), app.activity_as_ptr())
    };
    let entry = stage("entry load", unsafe { openxr::Entry::load(&platform_info) })?;
    let mut extensions = openxr::ExtensionSet::default();
    extensions.khr_android_create_instance = true;
    extensions.khr_vulkan_enable = true;
    let instance = stage(
        "create instance",
        entry.create_instance(
            &openxr::ApplicationInfo {
                application_name: "starling-xr-session",
                ..Default::default()
            },
            &extensions,
            &[],
            &platform_info,
        ),
    )?;
    let system = stage(
        "system",
        instance.system(openxr::FormFactor::HEAD_MOUNTED_DISPLAY),
    )?;

    // ---- Vulkan negotiation (legacy KHR_vulkan_enable) ----
    let requirements = stage(
        "graphics requirements",
        instance.graphics_requirements::<XrVulkan>(system),
    )?;
    log_info(&format!(
        "XR_SESSION: vulkan requirements min={} max={}",
        requirements.min_api_version_supported, requirements.max_api_version_supported
    ));

    let ash_entry = stage("vk entry", unsafe { ash::Entry::load() })?;
    let api_version = ash::vk::make_api_version(
        0,
        u32::from(requirements.min_api_version_supported.major()),
        u32::from(requirements.min_api_version_supported.minor()),
        requirements.min_api_version_supported.patch(),
    );
    let app_info = ash::vk::ApplicationInfo::default()
        .application_name(c"starling-xr-session")
        .api_version(api_version);
    let instance_exts = split_extensions(&stage(
        "vk instance extensions",
        instance.vulkan_legacy_instance_extensions(system),
    )?);
    let instance_ext_ptrs: Vec<*const c_char> =
        instance_exts.iter().map(|name| name.as_ptr()).collect();
    let vk_create_info = ash::vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&instance_ext_ptrs);
    let ash_instance = stage("vk create instance", unsafe {
        ash_entry.create_instance(&vk_create_info, None)
    })?;
    let vk_instance = ash_instance.handle();
    // SAFETY: vk_instance was just created above and outlives the session.
    let physical_device_raw = stage("vk graphics device", unsafe {
        instance.vulkan_graphics_device(system, vk_instance.as_raw() as *const c_void)
    })?;
    let physical_device = ash::vk::PhysicalDevice::from_raw(physical_device_raw as u64);
    let queue_family = {
        let families = unsafe {
            ash_instance.get_physical_device_queue_family_properties(physical_device)
        };
        families
            .iter()
            .position(|family| {
                family.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS)
            })
            .ok_or_else(|| "vk queue family: no graphics queue".to_owned())?
            as u32
    };
    let device_exts = split_extensions(&stage(
        "vk device extensions",
        instance.vulkan_legacy_device_extensions(system),
    )?);
    let device_ext_ptrs: Vec<*const c_char> =
        device_exts.iter().map(|name| name.as_ptr()).collect();
    let queue_priorities = [1.0f32];
    let queue_info = ash::vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&queue_priorities);
    let device_create_info = ash::vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_info))
        .enabled_extension_names(&device_ext_ptrs);
    let device = stage("vk create device", unsafe {
        ash_instance.create_device(physical_device, &device_create_info, None)
    })?;
    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    // ---- Session, reference space, swapchains ----
    // SAFETY: all four handles belong to the Vulkan objects created above,
    // which are kept alive in this scope until after the session drops.
    let (session, mut frame_waiter, mut frame_stream) = stage(
        "create session",
        unsafe {
            instance.create_session::<XrVulkan>(
                system,
                &openxr::vulkan::SessionCreateInfo {
                    instance: vk_instance.as_raw() as *const c_void,
                    physical_device: physical_device_raw,
                    device: device.handle().as_raw() as *const c_void,
                    queue_family_index: queue_family,
                    queue_index: 0,
                },
            )
        },
    )?;
    let space = stage(
        "reference space",
        session.create_reference_space(openxr::ReferenceSpaceType::LOCAL, openxr::Posef::IDENTITY),
    )?;
    let view_config = stage(
        "view configuration",
        instance.enumerate_view_configuration_views(
            system,
            openxr::ViewConfigurationType::PRIMARY_STEREO,
        ),
    )?;
    if view_config.len() < 2 {
        return Err(format!(
            "view configuration: expected 2 views, got {}",
            view_config.len()
        ));
    }
    let formats = stage("swapchain formats", session.enumerate_swapchain_formats())?;
    let format = pick_format(&formats);
    log_info(&format!(
        "XR_SESSION: swapchain format {format} ({} offered), views {}x{} / {}x{}",
        formats.len(),
        view_config[0].recommended_image_rect_width,
        view_config[0].recommended_image_rect_height,
        view_config[1].recommended_image_rect_width,
        view_config[1].recommended_image_rect_height,
    ));
    let mut swapchains = Vec::new();
    let mut swapchain_images = Vec::new();
    for view in view_config.iter().take(2) {
        let swapchain = stage(
            "create swapchain",
            session.create_swapchain(&openxr::SwapchainCreateInfo {
                create_flags: openxr::SwapchainCreateFlags::EMPTY,
                usage_flags: openxr::SwapchainUsageFlags::COLOR_ATTACHMENT
                    | openxr::SwapchainUsageFlags::TRANSFER_DST,
                format,
                sample_count: 1,
                width: view.recommended_image_rect_width,
                height: view.recommended_image_rect_height,
                face_count: 1,
                array_size: 1,
                mip_count: 1,
            }),
        )?;
        let images = stage("swapchain images", swapchain.enumerate_images())?;
        swapchains.push(swapchain);
        swapchain_images.push(images);
    }

    // ---- Vulkan clear machinery (one pool, one buffer, one fence) ----
    let command_pool = stage(
        "command pool",
        unsafe {
            device.create_command_pool(
                &ash::vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        },
    )?;
    let command_buffers = stage(
        "command buffers",
        unsafe {
            device.allocate_command_buffers(
                &ash::vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(ash::vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        },
    )?;
    let command_buffer = command_buffers[0];
    let fence = stage(
        "fence",
        unsafe { device.create_fence(&ash::vk::FenceCreateInfo::default(), None) },
    )?;

    // ---- Event + frame loop ----
    let started = Instant::now();
    let mut state = openxr::SessionState::UNKNOWN;
    let mut running = false;
    let mut focused = false;
    let mut frames: u64 = 0;
    let mut event_buffer = openxr::EventDataBuffer::new();
    let report = 'outer: loop {
        // Drain pending events first; they drive the state machine.
        loop {
            match instance.poll_event(&mut event_buffer) {
                Ok(Some(openxr::Event::SessionStateChanged(changed))) => {
                    state = changed.state();
                    log_info(&format!("XR_SESSION: state {state:?}"));
                    match state {
                        openxr::SessionState::READY => {
                            stage("begin session", session.begin(
                                openxr::ViewConfigurationType::PRIMARY_STEREO,
                            ))?;
                            running = true;
                        }
                        openxr::SessionState::STOPPING => {
                            let _ = session.end();
                            break 'outer format!(
                                "frames={frames} focused={focused} ended=stopping"
                            );
                        }
                        openxr::SessionState::EXITING
                        | openxr::SessionState::LOSS_PENDING => {
                            break 'outer format!(
                                "frames={frames} focused={focused} ended={state:?}"
                            );
                        }
                        openxr::SessionState::FOCUSED => focused = true,
                        _ => {}
                    }
                }
                Ok(Some(openxr::Event::InstanceLossPending(_))) => {
                    break 'outer format!(
                        "frames={frames} focused={focused} ended=instance_loss"
                    );
                }
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(error) => return Err(format!("poll event: {error}")),
            }
        }
        if !running {
            if started.elapsed() > READY_TIMEOUT {
                return Err(format!(
                    "ready timeout: session never became READY (last state {state:?})"
                ));
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
            continue;
        }
        if started.elapsed() > SESSION_CAP {
            log_info("XR_SESSION: session cap reached, requesting exit");
            let _ = session.request_exit();
            // The STOPPING event on the next iterations ends the loop;
            // bound the wait so we never hang here.
            if started.elapsed() > SESSION_CAP + std::time::Duration::from_secs(10) {
                break 'outer format!(
                    "frames={frames} focused={focused} ended=cap_no_stopping"
                );
            }
        }

        let frame_state = stage("wait frame", frame_waiter.wait())?;
        stage("begin frame", frame_stream.begin())?;
        if !frame_state.should_render {
            // An end() error must not kill the loop (see the pose note
            // below); log it and move on to the next frame.
            if let Err(error) = frame_stream.end(
                frame_state.predicted_display_time,
                openxr::EnvironmentBlendMode::OPAQUE,
                &[],
            ) {
                log_info(&format!(
                    "XR_SESSION: end frame failed (continuing): {error}"
                ));
            }
            continue;
        }
        let (view_flags, views) = stage(
            "locate views",
            session.locate_views(
                openxr::ViewConfigurationType::PRIMARY_STEREO,
                frame_state.predicted_display_time,
                &space,
            ),
        )?;
        if views.len() < 2 {
            if let Err(error) = frame_stream.end(
                frame_state.predicted_display_time,
                openxr::EnvironmentBlendMode::OPAQUE,
                &[],
            ) {
                log_info(&format!(
                    "XR_SESSION: end frame failed (continuing): {error}"
                ));
            }
            continue;
        }
        // When tracking drops for a frame, the view state flags lose the
        // valid bits and the crate substitutes default (zeroed) pose parts;
        // a zero quaternion is rejected by xrEndFrame with
        // XR_ERROR_POSE_INVALID (observed on-device at frame ~2160 of the
        // first 3a run, where it ended the loop). Submit an empty frame
        // instead of the projection layer.
        if !view_flags.contains(openxr::ViewStateFlags::POSITION_VALID)
            || !view_flags.contains(openxr::ViewStateFlags::ORIENTATION_VALID)
        {
            if let Err(error) = frame_stream.end(
                frame_state.predicted_display_time,
                openxr::EnvironmentBlendMode::OPAQUE,
                &[],
            ) {
                log_info(&format!(
                    "XR_SESSION: end frame failed (continuing): {error}"
                ));
            }
            continue;
        }

        // Clear both eyes into their swapchain images in one submission.
        unsafe {
            device
                .reset_command_pool(command_pool, ash::vk::CommandPoolResetFlags::empty())
                .map_err(|error| format!("reset command pool: {error}"))?;
            device
                .begin_command_buffer(
                    command_buffer,
                    &ash::vk::CommandBufferBeginInfo::default()
                        .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|error| format!("begin command buffer: {error}"))?;
        }
        for (eye, swapchain) in swapchains.iter_mut().enumerate() {
            let index = stage("acquire image", swapchain.acquire_image())? as usize;
            stage(
                "wait image",
                swapchain.wait_image(openxr::Duration::INFINITE),
            )?;
            let image = ash::vk::Image::from_raw(swapchain_images[eye][index]);
            let range = ash::vk::ImageSubresourceRange::default()
                .aspect_mask(ash::vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            unsafe {
                device.cmd_pipeline_barrier(
                    command_buffer,
                    ash::vk::PipelineStageFlags::ALL_COMMANDS,
                    ash::vk::PipelineStageFlags::TRANSFER,
                    ash::vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[ash::vk::ImageMemoryBarrier::default()
                        .old_layout(ash::vk::ImageLayout::UNDEFINED)
                        .new_layout(ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_access_mask(ash::vk::AccessFlags::empty())
                        .dst_access_mask(ash::vk::AccessFlags::TRANSFER_WRITE)
                        .image(image)
                        .subresource_range(range)],
                );
                device.cmd_clear_color_image(
                    command_buffer,
                    image,
                    ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &ash::vk::ClearColorValue {
                        float32: EYE_COLOURS[eye],
                    },
                    &[range],
                );
                device.cmd_pipeline_barrier(
                    command_buffer,
                    ash::vk::PipelineStageFlags::TRANSFER,
                    ash::vk::PipelineStageFlags::ALL_COMMANDS,
                    ash::vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[ash::vk::ImageMemoryBarrier::default()
                        .old_layout(ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .src_access_mask(ash::vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(ash::vk::AccessFlags::empty())
                        .image(image)
                        .subresource_range(range)],
                );
            }
        }
        unsafe {
            device
                .end_command_buffer(command_buffer)
                .map_err(|error| format!("end command buffer: {error}"))?;
            device
                .queue_submit(
                    queue,
                    &[ash::vk::SubmitInfo::default()
                        .command_buffers(std::slice::from_ref(&command_buffer))],
                    fence,
                )
                .map_err(|error| format!("queue submit: {error}"))?;
            device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|error| format!("wait fence: {error}"))?;
            device
                .reset_fences(&[fence])
                .map_err(|error| format!("reset fence: {error}"))?;
        }
        for swapchain in swapchains.iter_mut() {
            stage("release image", swapchain.release_image())?;
        }

        let projection_views = [
            openxr::CompositionLayerProjectionView::new()
                .pose(views[0].pose)
                .fov(views[0].fov)
                .sub_image(
                    openxr::SwapchainSubImage::new()
                        .swapchain(&swapchains[0])
                        .image_rect(openxr::Rect2Di {
                            offset: openxr::Offset2Di { x: 0, y: 0 },
                            extent: openxr::Extent2Di {
                                width: view_config[0].recommended_image_rect_width as i32,
                                height: view_config[0].recommended_image_rect_height as i32,
                            },
                        }),
                ),
            openxr::CompositionLayerProjectionView::new()
                .pose(views[1].pose)
                .fov(views[1].fov)
                .sub_image(
                    openxr::SwapchainSubImage::new()
                        .swapchain(&swapchains[1])
                        .image_rect(openxr::Rect2Di {
                            offset: openxr::Offset2Di { x: 0, y: 0 },
                            extent: openxr::Extent2Di {
                                width: view_config[1].recommended_image_rect_width as i32,
                                height: view_config[1].recommended_image_rect_height as i32,
                            },
                        }),
                ),
        ];
        let layer = openxr::CompositionLayerProjection::new()
            .space(&space)
            .views(&projection_views);
        if let Err(error) = frame_stream.end(
            frame_state.predicted_display_time,
            openxr::EnvironmentBlendMode::OPAQUE,
            &[&layer],
        ) {
            log_info(&format!(
                "XR_SESSION: end frame failed (continuing): {error}"
            ));
        }
        frames += 1;
        if frames % 120 == 0 {
            log_info(&format!(
                "XR_SESSION: frames={frames} state={state:?} focused={focused}"
            ));
        }
    };

    // ---- Teardown (reverse order; the session handles drop after) ----
    unsafe {
        let _ = device.device_wait_idle();
        device.destroy_fence(fence, None);
        device.destroy_command_pool(command_pool, None);
        device.destroy_device(None);
        ash_instance.destroy_instance(None);
    }
    Ok(report)
}

/// Split an OpenXR space-delimited extension list into owned C strings.
fn split_extensions(list: &str) -> Vec<CString> {
    list.split_whitespace()
        .filter_map(|name| CString::new(name).ok())
        .collect()
}

/// Prefer an sRGB format the runtime offered; fall back to its first choice.
/// (The format type is a private alias in the `openxr` crate, so this works
/// on the enumerated values by inference rather than naming the type.)
fn pick_format<T: Copy + PartialEq<u32>>(formats: &[T]) -> T {
    // VkFormat values (ash::vk::Format): R8G8B8A8_SRGB = 43,
    // R8G8B8A8_UNORM = 37, B8G8R8A8_SRGB = 50, B8G8R8A8_UNORM = 44. The
    // first 3a run fell back to the runtime's first offer, 22
    // (R8G8_SRGB — a two-channel format), because 44/50 were not offered;
    // the clear still displayed correctly, but 3b needs a real RGBA format.
    for preferred in [43u32, 37, 50, 44] {
        if let Some(found) = formats.iter().find(|format| **format == preferred) {
            return *found;
        }
    }
    formats.first().copied().unwrap_or_else(|| {
        // Unreachable in practice (a stereo system offers formats); the
        // fallback is never used because the caller checks beforehand.
        panic!("no swapchain formats offered")
    })
}
