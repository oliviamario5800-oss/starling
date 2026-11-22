//! Stage 2 VR spike, step 3b: the renderer's side of the OpenXR
//! bootstrap. The Vulkan/OpenXR work itself lives in the
//! `horizon-xr-boot` crate (this crate forbids unsafe code); here,
//! [`init`] forwards the Android entry's platform pointers, and
//! [`take_opened`] runs the bootstrap once and hands `Renderer::new`
//! the wrapped device as a surfaceless [`Opened`], with the eye's size
//! and format for the fabricated frame configuration. Any failure
//! falls back to the normal flat device search.

use std::sync::OnceLock;

use super::Opened;

/// Android entry hook: forward the JavaVM / activity pointers to the
/// bootstrap crate. A no-op off Android.
pub fn init(vm: *mut std::ffi::c_void, activity: *mut std::ffi::c_void) {
    horizon_xr_boot::init(vm, activity);
}

/// The bootstrap result, run at most once per process.
static BOOT: OnceLock<Option<horizon_xr_boot::Boot>> = OnceLock::new();

/// The `Opened` for `Renderer::new`, plus the eye configuration the
/// fabricated surface configuration must adopt. Runs the bootstrap on
/// first call; `None` means "use the normal device search".
pub(crate) fn take_opened() -> Option<(Opened, (u32, u32, wgpu::TextureFormat))> {
    let cache = BOOT
        .get_or_init(|| match horizon_xr_boot::run() {
            Ok(boot) => Some(boot),
            Err(error) => {
                log::warn!("XR_BOOT: unavailable: {error}");
                None
            }
        })
        .as_ref()?;
    Some((
        Opened {
            instance: cache.instance.clone(),
            surface: None,
            adapter: cache.adapter.clone(),
            device: cache.device.clone(),
            queue: cache.queue.clone(),
            device_lost: cache.device_lost.clone(),
            configured: None,
        },
        cache.eye,
    ))
}
