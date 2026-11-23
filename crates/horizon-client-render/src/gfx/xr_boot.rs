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
        let xr_frame = match ctx.begin_frame(&self.device, self.config.format, frame.title) {
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
        if !xr_frame.should_render {
            ctx.end_frame(xr_frame);
            return Some(Ok(true));
        }
        if xr_frame.menu.is_some() {
            return Some(self.draw_xr_menu(frame, ctx, xr_frame));
        }
        if xr_frame.eyes.len() != 2 {
            ctx.end_frame(xr_frame);
            return Some(Ok(true));
        }
        // A stereo frame ends any menu visit: the next menu places
        // its screen afresh in front of the user.
        *MENU_SCREEN.lock().unwrap_or_else(|error| error.into_inner()) = None;
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

    /// Draw a title frame as a virtual screen, Vivecraft-style: the
    /// frame is drawn flat into the menu swapchain's texture and the
    /// compositor shows that texture as a quad floating in front of
    /// the user, placed once per menu visit in the direction the user
    /// faces when it opens. A white patch marks where the user's gaze
    /// crosses the screen.
    #[cfg(target_os = "android")]
    fn draw_xr_menu(
        &mut self,
        frame: &super::Frame<'_>,
        ctx: &mut horizon_xr_boot::XrContext,
        mut xr_frame: horizon_xr_boot::XrFrame,
    ) -> std::result::Result<bool, Box<dyn std::error::Error>> {
        let menu_texture = xr_frame
            .menu
            .as_ref()
            .expect("menu frame")
            .texture
            .clone();
        // Head pose from the located views: position midway between
        // the eyes, orientation as located.
        let head = xr_frame.located_views[0].pose;
        let other = xr_frame.located_views[1].pose.position;
        let head_pos = glam::Vec3::new(
            (head.position.x + other.x) / 2.0,
            (head.position.y + other.y) / 2.0,
            (head.position.z + other.z) / 2.0,
        );
        let head_quat = glam::Quat::from_xyzw(
            head.orientation.x,
            head.orientation.y,
            head.orientation.z,
            head.orientation.w,
        );
        let screen = {
            let mut placement = MENU_SCREEN.lock().unwrap_or_else(|error| error.into_inner());
            if placement.is_none() {
                let forward = head_quat * glam::Vec3::NEG_Z;
                let flat = glam::Vec3::new(forward.x, 0.0, forward.z);
                // Looking nearly straight up or down leaves the yaw
                // undefined; fall back to the last placed yaw then.
                let yaw = if flat.length() > 0.15 {
                    let yaw = f32::atan2(-flat.x, -flat.z);
                    *MENU_LAST_YAW.lock().unwrap_or_else(|error| error.into_inner()) = yaw;
                    yaw
                } else {
                    *MENU_LAST_YAW.lock().unwrap_or_else(|error| error.into_inner())
                };
                let quat = glam::Quat::from_rotation_y(yaw);
                let dir = quat * glam::Vec3::NEG_Z;
                let width = 1.5f32;
                let height = width * (self.config.height as f32 / self.config.width as f32);
                let center = glam::Vec3::new(
                    head_pos.x + dir.x * 1.6,
                    head_pos.y,
                    head_pos.z + dir.z * 1.6,
                );
                *placement = Some(MenuScreen {
                    center,
                    quat,
                    width,
                    height,
                });
                log::info!("XR_MENU: screen placed, centre {center:?}, yaw {yaw:.2}");
            }
            placement.expect("just placed")
        };
        // The flat frame into the menu texture.
        let surface = self.surface.take();
        let depth = Self::depth(&self.device, &self.config);
        let offscreen = std::mem::replace(&mut self.offscreen, Some(menu_texture.clone()));
        let window_depth = std::mem::replace(&mut self.depth, depth);
        let result = self.draw_frame(frame, None, self.output);
        self.depth = window_depth;
        self.offscreen = offscreen;
        self.surface = surface;
        let drawn = match result {
            Ok(drawn) => drawn,
            Err(error) => {
                // The frame's images go back to the runtime only
                // once the device's work on them is done.
                self.wait_submissions();
                ctx.end_frame(xr_frame);
                return Err(error);
            }
        };
        // The eyes carry a cleared void under the menu quad.
        for eye in &xr_frame.eyes {
            let view = eye.texture.create_view(&wgpu::TextureViewDescriptor {
                format: Some(self.config.format.remove_srgb_suffix()),
                ..Default::default()
            });
            let mut encoder = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("XR menu void"),
                },
            );
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("XR menu void"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.008,
                            g: 0.008,
                            b: 0.016,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            self.queue.submit([encoder.finish()]);
        }
        // The gaze patch: where the head's forward ray crosses the
        // screen, copied in as a small white square.
        let basis = glam::Mat4::from_rotation_translation(screen.quat, screen.center);
        let ray_dir = head_quat * glam::Vec3::NEG_Z;
        let normal = screen.quat * glam::Vec3::Z;
        let denom = ray_dir.dot(normal);
        if denom.abs() > 1e-6 {
            let t = (screen.center - head_pos).dot(normal) / denom;
            if t > 0.0 {
                let hit = head_pos + ray_dir * t;
                let local = basis.inverse() * hit.extend(1.0);
                let u = local.x / screen.width + 0.5;
                let v = 0.5 - local.y / screen.height;
                if (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v) {
                    let patch = 24u32;
                    let x = ((u * self.config.width as f32) as i32 - patch as i32 / 2)
                        .clamp(0, (self.config.width - patch) as i32) as u32;
                    let y = ((v * self.config.height as f32) as i32 - patch as i32 / 2)
                        .clamp(0, (self.config.height - patch) as i32) as u32;
                    let white = white_patch(&self.device, &self.queue);
                    let mut encoder = self.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("XR menu gaze patch"),
                        },
                    );
                    encoder.copy_texture_to_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &white,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: &menu_texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d { x, y, z: 0 },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d {
                            width: patch,
                            height: patch,
                            depth_or_array_layers: 1,
                        },
                    );
                    self.queue.submit([encoder.finish()]);
                }
            }
        }
        xr_frame.menu_layer = Some((
            [screen.quat.x, screen.quat.y, screen.quat.z, screen.quat.w],
            [screen.center.x, screen.center.y, screen.center.z],
            [screen.width, screen.height],
        ));
        self.wait_submissions();
        ctx.end_frame(xr_frame);
        Ok(drawn)
    }
}

/// The menu screen's placement for the current menu visit, in the
/// OpenXR reference space: centre, upright orientation, size in
/// metres.
#[cfg(target_os = "android")]
#[derive(Clone, Copy)]
struct MenuScreen {
    center: glam::Vec3,
    quat: glam::Quat,
    width: f32,
    height: f32,
}

#[cfg(target_os = "android")]
static MENU_SCREEN: std::sync::Mutex<Option<MenuScreen>> = std::sync::Mutex::new(None);

/// The yaw of the most recent screen placement, for menu visits
/// that start with the user looking steeply up or down.
#[cfg(target_os = "android")]
static MENU_LAST_YAW: std::sync::Mutex<f32> = std::sync::Mutex::new(0.0);

/// A small all-white texture, created once, copied into the menu
/// texture each frame as the gaze patch.
#[cfg(target_os = "android")]
fn white_patch(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
    static WHITE: std::sync::OnceLock<wgpu::Texture> = std::sync::OnceLock::new();
    WHITE
        .get_or_init(|| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("XR gaze patch"),
                size: wgpu::Extent3d {
                    width: 32,
                    height: 32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &[255u8; 32 * 32 * 4],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(32 * 4),
                    rows_per_image: Some(32),
                },
                wgpu::Extent3d {
                    width: 32,
                    height: 32,
                    depth_or_array_layers: 1,
                },
            );
            texture
        })
        .clone()
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
    // perspective_rh with the near and far planes swapped, i.e. the
    // z terms are near/(far-near) and near*far/(far-near), so the
    // near plane maps to NDC 1 and the far plane to 0 (wgpu's
    // reversed-Z [0, 1] clip). An earlier version used the
    // OpenGL-style terms, which mapped the far plane to -1 and
    // clipped everything past mid-depth.
    let near = super::NEAR_PLANE;
    let far = frame.depth_far;
    let projection = glam::Mat4::from_cols(
        glam::Vec4::new(sx, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, sy, 0.0, 0.0),
        glam::Vec4::new(ox, oy, near / (far - near), -1.0),
        glam::Vec4::new(0.0, 0.0, near * far / (far - near), 0.0),
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
