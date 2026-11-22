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

impl super::Renderer {
    /// Draw `frame` into the headset instead of the window, when the XR
    /// session is live: bracket the frame with the OpenXR wait/begin and
    /// end calls and draw each eye into its swapchain texture, following
    /// `draw_scaled_frame`'s target-swap pattern. `None` means XR is not
    /// drawing (no session, not begun yet, or HDR is in the way) and the
    /// caller draws the flat frame as usual.
    #[cfg(not(target_os = "android"))]
    pub(crate) fn draw_xr_frame(
        &mut self,
        frame: &super::Frame<'_>,
    ) -> Option<std::result::Result<bool, Box<dyn std::error::Error>>> {
        let _ = frame;
        None
    }

    #[cfg(target_os = "android")]
    pub(crate) fn draw_xr_frame(
        &mut self,
        frame: &super::Frame<'_>,
    ) -> Option<std::result::Result<bool, Box<dyn std::error::Error>>> {
        if self.hdr.is_some() {
            // HDR's own targets and present path are flat-screen only.
            return None;
        }
        let mut guard = horizon_xr_boot::CTX
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let ctx = guard.as_mut()?;
        let xr_frame = match ctx.begin_frame(&self.device, self.config.format) {
            Ok(frame) => frame,
            Err(error) => {
                log::warn!("XR_FRAME: begin failed (drawing flat): {error}");
                return None;
            }
        };
        if !xr_frame.active {
            // The session has not begun yet; the flat frame shows.
            return None;
        }
        if !xr_frame.should_render || xr_frame.eyes.len() != 2 {
            ctx.end_frame(xr_frame);
            return Some(Ok(true));
        }
        let surface = self.surface.take();
        let mut drawn: std::result::Result<bool, Box<dyn std::error::Error>> = Ok(true);
        for eye in &xr_frame.eyes {
            let view = eye_view(frame, eye);
            let depth = Self::depth(&self.device, &self.config);
            let offscreen = std::mem::replace(&mut self.offscreen, Some(eye.texture.clone()));
            let window_depth = std::mem::replace(&mut self.depth, depth);
            self.view_override = Some(super::ViewSet::single(view));
            let result = self.draw_frame(frame, None, self.output);
            self.view_override = None;
            self.depth = window_depth;
            self.offscreen = offscreen;
            match result {
                Ok(true) => {}
                Ok(false) => drawn = Ok(false),
                Err(error) => {
                    drawn = Err(error);
                    break;
                }
            }
        }
        self.surface = surface;
        // The compositor may not have the images until the work is done.
        self.wait_submissions();
        ctx.end_frame(xr_frame);
        Some(drawn)
    }
}

/// One eye's [`super::View`] from its OpenXR pose and asymmetric field
/// of view, composed over the game's own look rotation: the headset's
/// rotation and offset apply in camera space, so head movement turns
/// the view relative to where the player faces. Units are metres,
/// which the world measures in blocks.
#[cfg(target_os = "android")]
fn eye_view(frame: &super::Frame<'_>, eye: &horizon_xr_boot::XrEyeTarget) -> super::View {
    let pose = eye.view.pose;
    let orientation = glam::Quat::from_xyzw(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    );
    let position = glam::Vec3::new(pose.position.x, pose.position.y, pose.position.z);
    let eye_view = glam::Mat4::from_rotation_translation(orientation, position).inverse();
    let view = eye_view * crate::camera::view_matrix(frame.yaw, frame.pitch);
    let fov = eye.view.fov;
    let (left, right, up, down) = (
        fov.angle_left.tan(),
        fov.angle_right.tan(),
        fov.angle_up.tan(),
        fov.angle_down.tan(),
    );
    let sx = 2.0 / (right - left);
    let sy = 2.0 / (up - down);
    let ox = (right + left) / (right - left);
    let oy = (up + down) / (up - down);
    // `world_projection` with the off-centre frustum: glam's
    // perspective_rh with the near and far planes swapped (reversed Z).
    let near = super::NEAR_PLANE;
    let far = frame.depth_far;
    let projection = glam::Mat4::from_cols(
        glam::Vec4::new(sx, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, sy, 0.0, 0.0),
        glam::Vec4::new(ox, oy, (near + far) / (far - near), -1.0),
        glam::Vec4::new(0.0, 0.0, 2.0 * near * far / (far - near), 0.0),
    );
    super::View {
        view,
        projection,
        aspect: (right - left) / (up - down),
        tan_half_fov: (up - down) / 2.0,
    }
}

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
