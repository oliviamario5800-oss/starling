//! Stage 2 VR spike, step 3b bootstrap: OpenXR owns the Vulkan instance
//! and device; wgpu wraps them (the Helio pattern). The wrapped device is
//! handed to the renderer as a surfaceless device (see
//! `horizon-client-render`'s `gfx::xr_boot`, the thin safe shim that calls
//! this crate — the render crate forbids unsafe code, like `horizon-hdr`
//! this crate exists to make the calls it can't).
//!
//! `horizon-client`'s Android entry calls [`init`] with the process's
//! JavaVM and activity pointers before the game starts. The first
//! `Renderer::new` that would otherwise search for a device runs [`run`]
//! (once, via the shim's cache): OpenXR instance, Vulkan negotiation,
//! the hal wrap, the session and its swapchains. Any failure is reported
//! and the caller falls back to the normal flat bootstrap — the XR path
//! must never brick the app.
//!
//! The session and swapchains live in [`CTX`] for the frame driver (the
//! render loop's XR bracketing, a later step). Nothing here begins the
//! session or submits a frame. Opt out with `HORIZON_NO_XR_BOOT`.
//!
//! Vulkan negotiation uses the legacy `XR_KHR_vulkan_enable` flow: the
//! `openxr` crate (0.22.0) wraps `create_vulkan_instance` for enable2 but
//! has no wrapper for `xrGetVulkanDeviceExtensions2KHR`, so the enable2
//! device-extension list is unreachable through its high-level API.

use std::sync::{Arc, OnceLock};
use std::sync::atomic::AtomicBool;

/// The JavaVM / activity pointers from the Android entry, as integers
/// (raw pointers are not `Send`; the values are process-lifetime).
static PLATFORM: OnceLock<(usize, usize)> = OnceLock::new();

/// Android entry hook: remember the platform pointers for the bootstrap.
/// A no-op off Android, when opted out, or outside the main process (the
/// `:launcher` process shares the entry and must not hold a session).
pub fn init(vm: *mut std::ffi::c_void, activity: *mut std::ffi::c_void) {
    #[cfg(target_os = "android")]
    {
        if std::env::var_os("HORIZON_NO_XR_BOOT").is_some() {
            log::info!("XR_BOOT: skipped (HORIZON_NO_XR_BOOT set)");
            return;
        }
        let cmdline = std::fs::read("/proc/self/cmdline").unwrap_or_default();
        if String::from_utf8_lossy(&cmdline).contains(':') {
            log::info!("XR_BOOT: skipped (not the main process)");
            return;
        }
        let _ = PLATFORM.set((vm as usize, activity as usize));
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = (vm, activity);
    }
}

/// The wrapped device the bootstrap produced.
pub struct Boot {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub device_lost: Arc<AtomicBool>,
    /// (width, height, format) of one eye's swapchain image.
    pub eye: (u32, u32, wgpu::TextureFormat),
}

/// Run the bootstrap. `Err` carries a stage-tagged reason; the caller
/// logs it and falls back to the flat device search.
#[cfg(not(target_os = "android"))]
pub fn run() -> Result<Boot, String> {
    Err("not Android".to_owned())
}

#[cfg(target_os = "android")]
pub fn run() -> Result<Boot, String> {
    imp::run()
}

#[cfg(target_os = "android")]
mod imp {
    use std::ffi::{c_char, c_void, CString};
    use std::sync::Mutex;

    use ash::vk::Handle as _;
    use openxr::vulkan::Vulkan as XrVulkan;

    use super::{Arc, AtomicBool, Boot, PLATFORM};

    /// The live XR session and its swapchains, created by the bootstrap
    /// and driven by the render loop's frame bracketing (a later step).
    pub struct XrContext {
        pub instance: openxr::Instance,
        pub session: openxr::Session<XrVulkan>,
        pub frame_waiter: openxr::FrameWaiter,
        pub frame_stream: openxr::FrameStream<XrVulkan>,
        pub swapchains: Vec<openxr::Swapchain<XrVulkan>>,
        pub images: Vec<Vec<u64>>,
        pub space: openxr::Space,
        pub views: Vec<openxr::ViewConfigurationView>,
        /// The swapchain format as a raw VkFormat value.
        pub format: u32,
        pub begun: bool,
        pub state: openxr::SessionState,
    }

    pub static CTX: Mutex<Option<XrContext>> = Mutex::new(None);

    type StageResult<T> = Result<T, String>;

    fn stage<T>(stage: &str, result: Result<T, impl std::fmt::Display>) -> StageResult<T> {
        result.map_err(|error| format!("{stage}: {error}"))
    }

    /// Split an OpenXR space-delimited extension list into owned C strings.
    fn split_extensions(list: &str) -> Vec<CString> {
        list.split_whitespace()
            .filter_map(|name| CString::new(name).ok())
            .collect()
    }

    /// The features the flat path would request (`open_adapter` in the
    /// render crate's surface_setup), minus ray queries: the spike device
    /// opens plain. Each helper there is `adapter.features() & mask`,
    /// except OIT's, which also needs its bounds format blendable.
    fn requested_features(adapter: &wgpu::Adapter) -> wgpu::Features {
        let supported = adapter.features();
        let mut features = supported
            & (wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
                | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
                | wgpu::Features::PIPELINE_STATISTICS_QUERY
                | wgpu::Features::SHADER_EARLY_DEPTH_TEST
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
                | wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
                | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
                | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                | wgpu::Features::RG11B10UFLOAT_RENDERABLE
                | wgpu::Features::SUBGROUP
                | wgpu::Features::VERTEX_WRITABLE_STORAGE
                | wgpu::Features::CLEAR_TEXTURE
                | wgpu::Features::TEXTURE_ATOMIC
                | wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::FLOAT32_FILTERABLE);
        // oit::features: FLOAT32_BLENDABLE only where Rgba32Float blends.
        if adapter
            .get_texture_format_features(wgpu::TextureFormat::Rgba32Float)
            .flags
            .contains(wgpu::TextureFormatFeatureFlags::BLENDABLE)
        {
            features |= supported & wgpu::Features::FLOAT32_BLENDABLE;
        }
        features
    }

    pub(super) fn run() -> StageResult<Boot> {
        let (vm, activity) = PLATFORM.get().copied().ok_or("init was not called")?;

        // ---- OpenXR instance + HMD system (same pattern as xr_session) ----
        // SAFETY: the pointers are the JavaVM and activity of this process,
        // kept alive by the GameActivity for the process lifetime.
        let platform_info = unsafe {
            openxr::AndroidPlatformInfo::new(vm as *mut c_void, activity as *mut c_void)
        };
        let entry = stage("entry load", unsafe { openxr::Entry::load(&platform_info) })?;
        let mut extensions = openxr::ExtensionSet::default();
        extensions.khr_android_create_instance = true;
        extensions.khr_vulkan_enable = true;
        let xr_instance = stage(
            "create instance",
            entry.create_instance(
                &openxr::ApplicationInfo {
                    application_name: "starling-xr-boot",
                    ..Default::default()
                },
                &extensions,
                &[],
                &platform_info,
            ),
        )?;
        let system = stage(
            "system",
            xr_instance.system(openxr::FormFactor::HEAD_MOUNTED_DISPLAY),
        )?;

        // ---- Vulkan negotiation (legacy KHR_vulkan_enable) ----
        let requirements = stage(
            "graphics requirements",
            xr_instance.graphics_requirements::<XrVulkan>(system),
        )?;
        log::info!(
            "XR_BOOT: vulkan requirements min={} max={}",
            requirements.min_api_version_supported,
            requirements.max_api_version_supported
        );
        let max = requirements.max_api_version_supported;
        let api_version = ash::vk::make_api_version(
            0,
            u32::from(max.major()),
            u32::from(max.minor()),
            max.patch(),
        );

        let ash_entry = stage("vk entry", unsafe { ash::Entry::load() })?;
        // The extensions wgpu-hal needs on the instance it wraps: the
        // Vulkan instance must be created with them, and the same list is
        // what `Instance::from_raw` is told about.
        let hal_flags = wgpu::InstanceFlags::default();
        let desired = wgpu::hal::vulkan::Instance::desired_extensions(
            &ash_entry,
            api_version,
            hal_flags,
        )
        .map_err(|error| format!("hal desired extensions: {error}"))?;

        let mut instance_exts = split_extensions(&stage(
            "vk instance extensions",
            xr_instance.vulkan_legacy_instance_extensions(system),
        )?);
        for name in &desired {
            let owned = CString::new(name.to_bytes()).expect("static extension name");
            if !instance_exts.iter().any(|existing| existing == &owned) {
                instance_exts.push(owned);
            }
        }
        let instance_ext_ptrs: Vec<*const c_char> =
            instance_exts.iter().map(|name| name.as_ptr()).collect();
        let app_info = ash::vk::ApplicationInfo::default()
            .application_name(c"starling-xr-boot")
            .api_version(api_version);
        let vk_create_info = ash::vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&instance_ext_ptrs);
        let ash_instance = stage("vk create instance", unsafe {
            ash_entry.create_instance(&vk_create_info, None)
        })?;
        let vk_instance = ash_instance.handle();
        // SAFETY: vk_instance was just created above and outlives the session.
        let physical_device_raw = stage("vk graphics device", unsafe {
            xr_instance.vulkan_graphics_device(system, vk_instance.as_raw() as *const c_void)
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

        // ---- Wrap the Vulkan objects into wgpu (the Helio pattern) ----
        // SAFETY: the instance was created from this entry with every
        // extension in `desired` enabled. The no-op drop callback keeps
        // wgpu from destroying the instance: teardown stays with the XR
        // side, which outlives the wgpu objects in program order.
        let hal_instance = unsafe {
            wgpu::hal::vulkan::Instance::from_raw(
                ash_entry.clone(),
                ash_instance.clone(),
                api_version,
                0,
                None,
                desired,
                hal_flags,
                wgpu::MemoryBudgetThresholds {
                    for_resource_creation: None,
                    for_device_loss: None,
                },
                false,
                Some(Box::new(|| {})),
            )
        }
        .map_err(|error| format!("hal instance from raw: {error}"))?;
        let exposed = hal_instance
            .expose_adapter(physical_device)
            .ok_or("hal expose adapter: none")?;
        // SAFETY: the hal instance and adapter come from the Vulkan
        // objects created above and are handed to wgpu whole.
        let wgpu_instance =
            unsafe { wgpu::Instance::from_hal::<wgpu::hal::api::Vulkan>(hal_instance) };
        let wgpu_adapter =
            unsafe { wgpu_instance.create_adapter_from_hal::<wgpu::hal::api::Vulkan>(exposed) };

        let features = requested_features(&wgpu_adapter);
        let limits = wgpu_adapter.limits();

        // SAFETY: the wgpu adapter wraps the hal adapter created above.
        let hal_adapter = unsafe { wgpu_adapter.as_hal::<wgpu::hal::api::Vulkan>() }
            .ok_or("hal adapter: none")?;
        let hal_exts = hal_adapter.required_device_extensions(features);
        let mut device_exts = split_extensions(&stage(
            "vk device extensions",
            xr_instance.vulkan_legacy_device_extensions(system),
        )?);
        for name in &hal_exts {
            let owned = CString::new(name.to_bytes()).expect("static extension name");
            if !device_exts.iter().any(|existing| existing == &owned) {
                device_exts.push(owned);
            }
        }
        let device_ext_ptrs: Vec<*const c_char> =
            device_exts.iter().map(|name| name.as_ptr()).collect();
        let mut hal_features = hal_adapter.physical_device_features(&hal_exts, features);
        let queue_priorities = [1.0f32];
        let queue_info = ash::vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&queue_priorities);
        let device_create_info = ash::vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_ext_ptrs);
        let device_create_info = hal_features.add_to_device_create(device_create_info);
        let ash_device = stage("vk create device", unsafe {
            ash_instance.create_device(physical_device, &device_create_info, None)
        })?;

        // SAFETY: the device was created from this adapter with the hal's
        // own feature struct and a superset of its required extensions.
        // The no-op drop callback keeps wgpu from destroying it.
        let open_device = unsafe {
            hal_adapter.device_from_raw(
                ash_device.clone(),
                Some(Box::new(|| {})),
                &hal_exts,
                features,
                &limits,
                &wgpu::MemoryHints::MemoryUsage,
                queue_family,
                0,
            )
        }
        .map_err(|error| format!("hal device from raw: {error}"))?;
        let desc = wgpu::DeviceDescriptor {
            label: Some("Horizon device"),
            required_features: features,
            required_limits: limits.clone(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            ..Default::default()
        };
        // SAFETY: the hal device comes from the adapter above.
        let (device, queue) = unsafe {
            wgpu_adapter.create_device_from_hal::<wgpu::hal::api::Vulkan>(open_device, &desc)
        }
        .map_err(|error| format!("wgpu device from hal: {error}"))?;
        let device_lost = Arc::new(AtomicBool::new(false));
        {
            let lost = device_lost.clone();
            device.set_device_lost_callback(move |reason, message| {
                if reason == wgpu::DeviceLostReason::Destroyed {
                    return;
                }
                lost.store(true, std::sync::atomic::Ordering::SeqCst);
                log::error!("XR_BOOT: device lost ({reason:?}): {message}");
            });
        }
        log::info!("XR_BOOT: wgpu device wrapped from the OpenXR Vulkan device");

        // ---- Session, reference space, swapchains ----
        // SAFETY: all four handles belong to the Vulkan objects created
        // above, which are kept alive in CTX / the boot cache.
        let (session, frame_waiter, frame_stream) = stage(
            "create session",
            unsafe {
                xr_instance.create_session::<XrVulkan>(
                    system,
                    &openxr::vulkan::SessionCreateInfo {
                        instance: vk_instance.as_raw() as *const c_void,
                        physical_device: physical_device_raw,
                        device: ash_device.handle().as_raw() as *const c_void,
                        queue_family_index: queue_family,
                        queue_index: 0,
                    },
                )
            },
        )?;
        let space = stage(
            "reference space",
            session.create_reference_space(
                openxr::ReferenceSpaceType::LOCAL,
                openxr::Posef::IDENTITY,
            ),
        )?;
        let view_config = stage(
            "view configuration",
            xr_instance.enumerate_view_configuration_views(
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
        log::info!(
            "XR_BOOT: swapchain formats offered: {}",
            formats
                .iter()
                .take(16)
                .map(|format| format.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let (format, wgpu_format) = pick_format(&formats)?;
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
        let eye = (
            view_config[0].recommended_image_rect_width,
            view_config[0].recommended_image_rect_height,
            wgpu_format,
        );
        log::info!(
            "XR_BOOT: session ready, format {format} ({wgpu_format:?}), views {}x{} / {}x{}",
            view_config[0].recommended_image_rect_width,
            view_config[0].recommended_image_rect_height,
            view_config[1].recommended_image_rect_width,
            view_config[1].recommended_image_rect_height,
        );
        *CTX.lock().unwrap_or_else(|error| error.into_inner()) = Some(XrContext {
            instance: xr_instance,
            session,
            frame_waiter,
            frame_stream,
            swapchains,
            images: swapchain_images,
            space,
            views: view_config,
            format,
            begun: false,
            state: openxr::SessionState::UNKNOWN,
        });
        Ok(Boot {
            instance: wgpu_instance,
            adapter: wgpu_adapter,
            device,
            queue,
            device_lost,
            eye,
        })
    }

    /// Prefer an sRGB RGBA format the renderer can draw into; the runtime
    /// offers dozens (its first is a two-channel format the clear test
    /// tolerated but wgpu has no render target for).
    fn pick_format(formats: &[u32]) -> StageResult<(u32, wgpu::TextureFormat)> {
        use ash::vk::Format;
        let candidates = [
            (Format::R8G8B8A8_SRGB, wgpu::TextureFormat::Rgba8UnormSrgb),
            (Format::R8G8B8A8_UNORM, wgpu::TextureFormat::Rgba8Unorm),
            (Format::B8G8R8A8_SRGB, wgpu::TextureFormat::Bgra8UnormSrgb),
            (Format::B8G8R8A8_UNORM, wgpu::TextureFormat::Bgra8Unorm),
        ];
        for (vk, wgpu) in candidates {
            let raw = vk.as_raw() as u32;
            if formats.contains(&raw) {
                return Ok((raw, wgpu));
            }
        }
        Err(format!(
            "swapchain formats: no RGBA format among {} offered",
            formats.len()
        ))
    }
}

/// The live XR context for the frame driver (Android only; the render
/// loop's XR bracketing locks it once per frame).
#[cfg(target_os = "android")]
pub use imp::{CTX, XrContext};
