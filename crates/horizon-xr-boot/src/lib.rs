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

/// The latest controller input, published by the XR frame driver
/// every frame (Stage 3) and read by the game's pad polling. Plain
/// data in the game's terms: sticks are `[x, y]` with right and up
/// positive, triggers run 0..1, the face buttons are named by their
/// position (south = the Quest controller's A, east = B, west = X,
/// north = Y). `menu_uv` / `menu_trigger` are the render shim's to
/// fill on menu frames (it owns the screen's placement); the frame
/// driver preserves them across its per-frame overwrite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InputSnapshot {
    /// The session is running and actions synced this frame.
    pub active: bool,
    pub left_stick: [f32; 2],
    pub right_stick: [f32; 2],
    pub left_trigger: f32,
    pub right_trigger: f32,
    pub south: bool,
    pub east: bool,
    pub west: bool,
    pub north: bool,
    /// The left controller's menu button.
    pub menu: bool,
    pub left_stick_click: bool,
    pub right_stick_click: bool,
    /// Aim poses in the reference space: (quaternion xyzw, position).
    pub aim_left: Option<([f32; 4], [f32; 3])>,
    pub aim_right: Option<([f32; 4], [f32; 3])>,
    /// Where the aim ray meets the menu screen: 0..1 over the menu
    /// texture, y down. Set by the render shim on menu frames.
    pub menu_uv: Option<[f32; 2]>,
    /// The aiming hand's trigger value, for menu clicks.
    pub menu_trigger: f32,
}

impl InputSnapshot {
    pub const fn empty() -> Self {
        Self {
            active: false,
            left_stick: [0.0; 2],
            right_stick: [0.0; 2],
            left_trigger: 0.0,
            right_trigger: 0.0,
            south: false,
            east: false,
            west: false,
            north: false,
            menu: false,
            left_stick_click: false,
            right_stick_click: false,
            aim_left: None,
            aim_right: None,
            menu_uv: None,
            menu_trigger: 0.0,
        }
    }
}

impl Default for InputSnapshot {
    fn default() -> Self {
        Self::empty()
    }
}

/// The shared input snapshot (see [`InputSnapshot`]).
pub static INPUT: std::sync::Mutex<InputSnapshot> = std::sync::Mutex::new(InputSnapshot::empty());

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
        /// Swapchain for the menu screen, submitted as a compositor
        /// quad layer instead of drawn into the eyes.
        pub menu_swapchain: openxr::Swapchain<XrVulkan>,
        pub menu_images: Vec<u64>,
        pub menu_textures: Vec<Option<wgpu::Texture>>,
        pub space: openxr::Space,
        pub views: Vec<openxr::ViewConfigurationView>,
        /// The swapchain format as a raw VkFormat value.
        pub format: u32,
        pub begun: bool,
        pub state: openxr::SessionState,
        /// Wrapped swapchain images, per eye per image index, created on
        /// first acquire and reused for the session's lifetime.
        pub textures: [Vec<Option<wgpu::Texture>>; 2],
        /// The controller actions (Stage 3); None when their setup
        /// failed — input is an addition, never a boot requirement.
        pub actions: Option<XrActions>,
        /// A sync failure was already logged (don't spam per frame).
        pub input_warned: bool,
    }

    /// The OpenXR action set for the Touch controllers and the aim
    /// spaces derived from it. One action per hand and control (no
    /// subaction paths): the bindings name the hand's path directly.
    pub struct XrActions {
        pub set: openxr::ActionSet,
        pub left_stick: openxr::Action<openxr::Vector2f>,
        pub right_stick: openxr::Action<openxr::Vector2f>,
        pub left_trigger: openxr::Action<f32>,
        pub right_trigger: openxr::Action<f32>,
        pub south: openxr::Action<bool>,
        pub east: openxr::Action<bool>,
        pub west: openxr::Action<bool>,
        pub north: openxr::Action<bool>,
        pub menu: openxr::Action<bool>,
        pub left_click: openxr::Action<bool>,
        pub right_click: openxr::Action<bool>,
        pub aim_left_space: openxr::Space,
        pub aim_right_space: openxr::Space,
    }

    /// One eye's render target for a begun frame.
    pub struct XrEyeTarget {
        pub texture: wgpu::Texture,
        pub view: openxr::View,
        pub image_index: u32,
    }

    /// The menu screen's render target for a begun frame: the flat
    /// frame is drawn into it and the compositor shows it as a quad.
    pub struct XrMenuTarget {
        pub texture: wgpu::Texture,
        pub image_index: u32,
    }

    /// A begun XR frame. `active` is false when the session has not
    /// begun yet: nothing was waited on and [`XrContext::end_frame`]
    /// must not be called for it (it is a no-op if it is).
    pub struct XrFrame {
        pub active: bool,
        pub should_render: bool,
        pub views_valid: bool,
        pub display_time: openxr::Time,
        pub eyes: Vec<XrEyeTarget>,
        pub extent: openxr::Extent2Di,
        /// The located eye views, filled for menu frames too (their
        /// poses place the screen and the gaze cursor).
        pub located_views: Vec<openxr::View>,
        /// The menu target, when this frame was begun as a menu frame.
        pub menu: Option<XrMenuTarget>,
        /// Where the compositor should put the menu screen this frame:
        /// (orientation quaternion xyzw, position, size in metres),
        /// the pose in the reference space. The renderer fills this in
        /// before [`XrContext::end_frame`]; without it a menu frame
        /// submits no layer.
        pub menu_layer: Option<([f32; 4], [f32; 3], [f32; 2])>,
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
        // Controller actions (Stage 3): a failure here only costs
        // input, so it is logged, not fatal.
        let actions = match create_actions(&xr_instance, &session) {
            Ok(actions) => {
                log::info!("XR_BOOT: controller actions ready");
                Some(actions)
            }
            Err(error) => {
                log::warn!("XR_BOOT: controller actions unavailable: {error}");
                None
            }
        };
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
                        | openxr::SwapchainUsageFlags::TRANSFER_DST
                        | openxr::SwapchainUsageFlags::TRANSFER_SRC,
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
        // The menu screen's swapchain: same size and format as an eye,
        // shown by the compositor as a quad layer on menu frames.
        let menu_swapchain = stage(
            "create menu swapchain",
            session.create_swapchain(&openxr::SwapchainCreateInfo {
                create_flags: openxr::SwapchainCreateFlags::EMPTY,
                usage_flags: openxr::SwapchainUsageFlags::COLOR_ATTACHMENT
                    | openxr::SwapchainUsageFlags::TRANSFER_DST
                    | openxr::SwapchainUsageFlags::TRANSFER_SRC,
                format,
                sample_count: 1,
                width: view_config[0].recommended_image_rect_width,
                height: view_config[0].recommended_image_rect_height,
                face_count: 1,
                array_size: 1,
                mip_count: 1,
            }),
        )?;
        let menu_images = stage("menu swapchain images", menu_swapchain.enumerate_images())?;
        let menu_textures = vec![None; menu_images.len()];
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
        let textures = [
            vec![None; swapchain_images[0].len()],
            vec![None; swapchain_images[1].len()],
        ];
        *CTX.lock().unwrap_or_else(|error| error.into_inner()) = Some(XrContext {
            instance: xr_instance,
            session,
            frame_waiter,
            frame_stream,
            swapchains,
            images: swapchain_images,
            menu_swapchain,
            menu_images,
            menu_textures,
            space,
            views: view_config,
            format,
            begun: false,
            state: openxr::SessionState::UNKNOWN,
            textures,
            actions,
            input_warned: false,
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

    /// Build the Touch-controller action set, suggest its bindings
    /// for the Oculus Touch profile and attach it to the session
    /// (Stage 3 input). One action per hand and control, no subaction
    /// paths: each binding names its hand's path directly.
    fn create_actions(
        instance: &openxr::Instance,
        session: &openxr::Session<XrVulkan>,
    ) -> StageResult<XrActions> {
        let set = stage(
            "action set",
            instance.create_action_set("gameplay", "Gameplay", 0),
        )?;
        let left_stick = stage(
            "action",
            set.create_action::<openxr::Vector2f>("left_stick", "Left stick", &[]),
        )?;
        let right_stick = stage(
            "action",
            set.create_action::<openxr::Vector2f>("right_stick", "Right stick", &[]),
        )?;
        let left_trigger = stage(
            "action",
            set.create_action::<f32>("left_trigger", "Left trigger", &[]),
        )?;
        let right_trigger = stage(
            "action",
            set.create_action::<f32>("right_trigger", "Right trigger", &[]),
        )?;
        let south = stage("action", set.create_action::<bool>("south", "A", &[]))?;
        let east = stage("action", set.create_action::<bool>("east", "B", &[]))?;
        let west = stage("action", set.create_action::<bool>("west", "X", &[]))?;
        let north = stage("action", set.create_action::<bool>("north", "Y", &[]))?;
        let menu = stage("action", set.create_action::<bool>("menu", "Menu", &[]))?;
        let left_click = stage(
            "action",
            set.create_action::<bool>("left_click", "Left stick click", &[]),
        )?;
        let right_click = stage(
            "action",
            set.create_action::<bool>("right_click", "Right stick click", &[]),
        )?;
        let aim_left = stage(
            "action",
            set.create_action::<openxr::Posef>("aim_left", "Left aim", &[]),
        )?;
        let aim_right = stage(
            "action",
            set.create_action::<openxr::Posef>("aim_right", "Right aim", &[]),
        )?;
        let path = |name: &str| stage("action path", instance.string_to_path(name));
        let bindings = [
            openxr::Binding::new(&left_stick, path("/user/hand/left/input/thumbstick")?),
            openxr::Binding::new(&right_stick, path("/user/hand/right/input/thumbstick")?),
            openxr::Binding::new(&left_trigger, path("/user/hand/left/input/trigger/value")?),
            openxr::Binding::new(&right_trigger, path("/user/hand/right/input/trigger/value")?),
            openxr::Binding::new(&south, path("/user/hand/right/input/a/click")?),
            openxr::Binding::new(&east, path("/user/hand/right/input/b/click")?),
            openxr::Binding::new(&west, path("/user/hand/left/input/x/click")?),
            openxr::Binding::new(&north, path("/user/hand/left/input/y/click")?),
            openxr::Binding::new(&menu, path("/user/hand/left/input/menu/click")?),
            openxr::Binding::new(&left_click, path("/user/hand/left/input/thumbstick/click")?),
            openxr::Binding::new(&right_click, path("/user/hand/right/input/thumbstick/click")?),
            openxr::Binding::new(&aim_left, path("/user/hand/left/input/aim/pose")?),
            openxr::Binding::new(&aim_right, path("/user/hand/right/input/aim/pose")?),
        ];
        stage(
            "suggest bindings",
            instance.suggest_interaction_profile_bindings(
                path("/interaction_profiles/oculus/touch_controller")?,
                &bindings,
            ),
        )?;
        stage("attach actions", session.attach_action_sets(&[&set]))?;
        let aim_left_space = stage(
            "aim space",
            aim_left.create_space(session, openxr::Path::NULL, openxr::Posef::IDENTITY),
        )?;
        let aim_right_space = stage(
            "aim space",
            aim_right.create_space(session, openxr::Path::NULL, openxr::Posef::IDENTITY),
        )?;
        Ok(XrActions {
            set,
            left_stick,
            right_stick,
            left_trigger,
            right_trigger,
            south,
            east,
            west,
            north,
            menu,
            left_click,
            right_click,
            aim_left_space,
            aim_right_space,
        })
    }

    impl XrContext {
        /// Mark the published input inactive (Stage 3): the game's
        /// polling then merges nothing, so no stale stick or button
        /// can stay held while the session is unfocused or not
        /// rendering.
        fn input_inactive(&mut self) {
            super::INPUT
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .active = false;
        }

        /// Sync the controller actions and publish the snapshot the
        /// game's pad polling reads (Stage 3). Errors are logged once;
        /// input then simply stays inactive.
        fn sync_input(&mut self, time: openxr::Time) {
            let Some(actions) = &self.actions else {
                return;
            };
            if let Err(error) = self
                .session
                .sync_actions(&[openxr::ActiveActionSet::new(&actions.set)])
            {
                if !self.input_warned {
                    self.input_warned = true;
                    log::warn!("XR_INPUT: sync failed: {error}");
                }
                // A stale snapshot would merge as held input forever.
                self.input_inactive();
                return;
            }
            let session = &self.session;
            let space = &self.space;
            let null = openxr::Path::NULL;
            let vec2 = |action: &openxr::Action<openxr::Vector2f>| {
                action
                    .state(session, null)
                    .map(|state| {
                        if state.is_active {
                            [state.current_state.x, state.current_state.y]
                        } else {
                            [0.0; 2]
                        }
                    })
                    .unwrap_or([0.0; 2])
            };
            let analog = |action: &openxr::Action<f32>| {
                action
                    .state(session, null)
                    .map(|state| if state.is_active { state.current_state } else { 0.0 })
                    .unwrap_or(0.0)
            };
            let button = |action: &openxr::Action<bool>| {
                action
                    .state(session, null)
                    .map(|state| state.is_active && state.current_state)
                    .unwrap_or(false)
            };
            let aim = |aim_space: &openxr::Space| {
                aim_space.locate(space, time).ok().and_then(|location| {
                    let valid = openxr::SpaceLocationFlags::POSITION_VALID
                        | openxr::SpaceLocationFlags::ORIENTATION_VALID;
                    if location.location_flags.contains(valid) {
                        let pose = location.pose;
                        Some((
                            [
                                pose.orientation.x,
                                pose.orientation.y,
                                pose.orientation.z,
                                pose.orientation.w,
                            ],
                            [pose.position.x, pose.position.y, pose.position.z],
                        ))
                    } else {
                        None
                    }
                })
            };
            let mut snapshot = super::InputSnapshot {
                active: true,
                left_stick: vec2(&actions.left_stick),
                right_stick: vec2(&actions.right_stick),
                left_trigger: analog(&actions.left_trigger),
                right_trigger: analog(&actions.right_trigger),
                south: button(&actions.south),
                east: button(&actions.east),
                west: button(&actions.west),
                north: button(&actions.north),
                menu: button(&actions.menu),
                left_stick_click: button(&actions.left_click),
                right_stick_click: button(&actions.right_click),
                aim_left: aim(&actions.aim_left_space),
                aim_right: aim(&actions.aim_right_space),
                ..super::InputSnapshot::empty()
            };
            let mut guard = super::INPUT
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            // The menu fields belong to the render shim.
            snapshot.menu_uv = guard.menu_uv;
            snapshot.menu_trigger = guard.menu_trigger;
            *guard = snapshot;
        }

        /// Drain pending session events, beginning the session when the
        /// runtime offers it. Returns whether the session is begun.
        pub fn poll_events(&mut self) -> StageResult<bool> {
            // A fresh buffer per drain: EventDataBuffer is not Send, so
            // it cannot live in the shared context.
            let mut event_buffer = openxr::EventDataBuffer::new();
            loop {
                match self.instance.poll_event(&mut event_buffer) {
                    Ok(Some(openxr::Event::SessionStateChanged(changed))) => {
                        self.state = changed.state();
                        log::info!("XR_FRAME: state {:?}", self.state);
                        match self.state {
                            openxr::SessionState::READY => {
                                stage(
                                    "begin session",
                                    self.session.begin(
                                        openxr::ViewConfigurationType::PRIMARY_STEREO,
                                    ),
                                )?;
                                self.begun = true;
                            }
                            openxr::SessionState::STOPPING => {
                                let _ = self.session.end();
                                self.begun = false;
                            }
                            _ => {}
                        }
                    }
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(error) => {
                        log::warn!("XR_FRAME: poll event failed (continuing): {error}");
                        break;
                    }
                }
            }
            Ok(self.begun)
        }

        /// Wait for the next frame, begin it, locate the eye views and
        /// acquire the frame's render targets, wrapped as wgpu
        /// textures: both eye images for a stereo frame, or the menu
        /// swapchain's image when `want_menu` (the eyes are located
        /// but not acquired then). A frame that should not render (or
        /// whose poses are invalid) comes back with `should_render`
        /// false and no targets; end it with [`Self::end_frame`] all
        /// the same.
        pub fn begin_frame(
            &mut self,
            device: &wgpu::Device,
            wgpu_format: wgpu::TextureFormat,
            want_menu: bool,
        ) -> StageResult<XrFrame> {
            let extent = openxr::Extent2Di {
                width: self.views[0].recommended_image_rect_width as i32,
                height: self.views[0].recommended_image_rect_height as i32,
            };
            let skip = XrFrame {
                active: false,
                should_render: false,
                views_valid: false,
                display_time: openxr::Time::from_nanos(0),
                eyes: Vec::new(),
                extent,
                located_views: Vec::new(),
                menu: None,
                menu_layer: None,
            };
            if !self.poll_events()? {
                self.input_inactive();
                return Ok(skip);
            }
            let frame_state = stage("wait frame", self.frame_waiter.wait())?;
            stage("begin frame", self.frame_stream.begin())?;
            let mut frame = XrFrame {
                active: true,
                ..skip
            };
            frame.display_time = frame_state.predicted_display_time;
            if !frame_state.should_render {
                self.input_inactive();
                return Ok(frame);
            }
            let (flags, views) = stage(
                "locate views",
                self.session.locate_views(
                    openxr::ViewConfigurationType::PRIMARY_STEREO,
                    frame_state.predicted_display_time,
                    &self.space,
                ),
            )?;
            frame.views_valid = flags.contains(openxr::ViewStateFlags::POSITION_VALID)
                && flags.contains(openxr::ViewStateFlags::ORIENTATION_VALID);
            if !frame.views_valid || views.len() < 2 {
                return Ok(frame);
            }
            frame.should_render = true;
            frame.located_views = views.clone();
            self.sync_input(frame_state.predicted_display_time);
            if want_menu {
                // Menu frames acquire the menu image AND both eye
                // images: the eyes carry a cleared projection layer
                // under the menu quad, so the compositor always gets
                // the base layer it expects.
                let index = stage("acquire menu image", self.menu_swapchain.acquire_image())?;
                stage(
                    "wait menu image",
                    self.menu_swapchain.wait_image(openxr::Duration::INFINITE),
                )?;
                if self.menu_textures[index as usize].is_none() {
                    let texture = wrap_swapchain_image(
                        device,
                        self.menu_images[index as usize],
                        wgpu_format,
                        extent,
                    )?;
                    self.menu_textures[index as usize] = Some(texture);
                }
                let texture = self.menu_textures[index as usize]
                    .as_ref()
                    .expect("just wrapped")
                    .clone();
                frame.menu = Some(XrMenuTarget {
                    texture,
                    image_index: index,
                });
            }
            for (eye, view) in views.iter().take(2).enumerate() {
                let index = stage("acquire image", self.swapchains[eye].acquire_image())?;
                stage(
                    "wait image",
                    self.swapchains[eye].wait_image(openxr::Duration::INFINITE),
                )?;
                if self.textures[eye][index as usize].is_none() {
                    let texture = wrap_swapchain_image(
                        device,
                        self.images[eye][index as usize],
                        wgpu_format,
                        extent,
                    )?;
                    self.textures[eye][index as usize] = Some(texture);
                }
                let texture = self.textures[eye][index as usize]
                    .as_ref()
                    .expect("just wrapped")
                    .clone();
                frame.eyes.push(XrEyeTarget {
                    texture,
                    view: *view,
                    image_index: index,
                });
            }
            Ok(frame)
        }

        /// Release the frame's images and submit it to the compositor.
        /// Call only after the device's submissions for the frame have
        /// completed. End-frame errors are logged, never fatal (a pose
        /// can go invalid mid-frame; the next frame recovers).
        pub fn end_frame(&mut self, frame: XrFrame) {
            if !frame.active {
                return;
            }
            if frame.menu.is_some() {
                if let Err(error) = self.menu_swapchain.release_image() {
                    log::warn!("XR_FRAME: release menu image failed (continuing): {error}");
                }
            }
            for (eye, _) in frame.eyes.iter().enumerate() {
                if let Err(error) = self.swapchains[eye].release_image() {
                    log::warn!("XR_FRAME: release image {eye} failed (continuing): {error}");
                }
            }
            // The menu quad (built inline in the branches that submit
            // it; the layer types borrow the swapchain handles).
            macro_rules! menu_quad {
                ($orientation:expr, $position:expr, $size:expr, $extent:expr) => {
                    openxr::CompositionLayerQuad::new()
                        .space(&self.space)
                        .eye_visibility(openxr::EyeVisibility::BOTH)
                        .sub_image(
                            openxr::SwapchainSubImage::new()
                                .swapchain(&self.menu_swapchain)
                                .image_rect(openxr::Rect2Di {
                                    offset: openxr::Offset2Di { x: 0, y: 0 },
                                    extent: $extent,
                                }),
                        )
                        .pose(openxr::Posef {
                            orientation: openxr::Quaternionf {
                                x: $orientation[0],
                                y: $orientation[1],
                                z: $orientation[2],
                                w: $orientation[3],
                            },
                            position: openxr::Vector3f {
                                x: $position[0],
                                y: $position[1],
                                z: $position[2],
                            },
                        })
                        .size(openxr::Extent2Df {
                            width: $size[0],
                            height: $size[1],
                        })
                };
            }
            let stereo_ok = frame.should_render && frame.views_valid && frame.eyes.len() == 2;
            let result = if stereo_ok {
                let proj_views: Vec<openxr::CompositionLayerProjectionView<'_, XrVulkan>> = frame
                    .eyes
                    .iter()
                    .enumerate()
                    .map(|(eye, target)| {
                        openxr::CompositionLayerProjectionView::new()
                            .pose(target.view.pose)
                            .fov(target.view.fov)
                            .sub_image(
                                openxr::SwapchainSubImage::new()
                                    .swapchain(&self.swapchains[eye])
                                    .image_rect(openxr::Rect2Di {
                                        offset: openxr::Offset2Di { x: 0, y: 0 },
                                        extent: frame.extent,
                                    }),
                            )
                    })
                    .collect();
                let layer = openxr::CompositionLayerProjection::new()
                    .space(&self.space)
                    .views(&proj_views);
                match frame.menu_layer {
                    // Menu frame: the cleared eyes as the base layer,
                    // the screen quad on top.
                    Some((orientation, position, size)) => {
                        let quad = menu_quad!(orientation, position, size, frame.extent);
                        self.frame_stream.end(
                            frame.display_time,
                            openxr::EnvironmentBlendMode::OPAQUE,
                            &[&layer, &quad],
                        )
                    }
                    None => self.frame_stream.end(
                        frame.display_time,
                        openxr::EnvironmentBlendMode::OPAQUE,
                        &[&layer],
                    ),
                }
            } else if frame.should_render && frame.views_valid {
                match frame.menu_layer {
                    Some((orientation, position, size)) => {
                        let quad = menu_quad!(orientation, position, size, frame.extent);
                        self.frame_stream.end(
                            frame.display_time,
                            openxr::EnvironmentBlendMode::OPAQUE,
                            &[&quad],
                        )
                    }
                    None => self.frame_stream.end(
                        frame.display_time,
                        openxr::EnvironmentBlendMode::OPAQUE,
                        &[],
                    ),
                }
            } else {
                self.frame_stream.end(
                    frame.display_time,
                    openxr::EnvironmentBlendMode::OPAQUE,
                    &[],
                )
            };
            if let Err(error) = result {
                log::warn!("XR_FRAME: end frame failed (continuing): {error}");
            }
        }
    }

    /// Wrap one swapchain `VkImage` as a wgpu texture wgpu does not own:
    /// with external memory and a no-op drop callback it destroys
    /// neither the image nor its memory.
    fn wrap_swapchain_image(
        device: &wgpu::Device,
        vk_image: u64,
        format: wgpu::TextureFormat,
        extent: openxr::Extent2Di,
    ) -> StageResult<wgpu::Texture> {
        use ash::vk::Handle as _;
        let size = wgpu::Extent3d {
            width: extent.width as u32,
            height: extent.height as u32,
            depth_or_array_layers: 1,
        };
        // The renderer draws into its output through an sRGB-stripped
        // view (its output format is the surface format minus the sRGB
        // suffix), so the wrapped image must declare both spellings or
        // the first frame's view creation fails validation.
        let stripped = format.remove_srgb_suffix();
        let mut view_formats = vec![format];
        if stripped != format {
            view_formats.push(stripped);
        }
        // SAFETY: the device wraps the hal device the session was
        // created on (the bootstrap built both from the same handles).
        let hal_device = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }
            .ok_or("hal device: none")?;
        let hal_desc = wgpu::hal::TextureDescriptor {
            label: Some("XR eye"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUses::COLOR_TARGET
                | wgpu::TextureUses::COPY_SRC
                | wgpu::TextureUses::COPY_DST,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: view_formats.clone(),
        };
        // SAFETY: the image belongs to this session's swapchain, is
        // acquired and waited on, and outlives the returned texture
        // (textures are cached for the session's lifetime).
        let hal_texture = unsafe {
            hal_device.texture_from_raw(
                ash::vk::Image::from_raw(vk_image),
                &hal_desc,
                Some(Box::new(|| {})),
                wgpu::hal::vulkan::TextureMemory::External,
            )
        };
        // SAFETY: the hal texture was just made from this device with a
        // matching descriptor; the image starts undefined to wgpu.
        let texture = unsafe {
            device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(
                hal_texture,
                &wgpu::TextureDescriptor {
                    label: Some("XR eye"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &view_formats,
                },
                wgpu::TextureUses::UNINITIALIZED,
            )
        };
        Ok(texture)
    }
}

/// The live XR context for the frame driver (Android only; the render
/// loop's XR bracketing locks it once per frame).
#[cfg(target_os = "android")]
pub use imp::{CTX, XrContext, XrEyeTarget, XrFrame, XrMenuTarget};
