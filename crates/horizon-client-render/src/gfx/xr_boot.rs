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

/// The XR controller input snapshot type, re-exported for the
/// game's pad polling.
pub use horizon_xr_boot::InputSnapshot;

/// The latest XR controller input, for the game's pad polling
/// (Stage 3). `None` when no XR session is publishing input.
pub fn input_snapshot() -> Option<horizon_xr_boot::InputSnapshot> {
    #[cfg(target_os = "android")]
    {
        let snapshot = *horizon_xr_boot::INPUT
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        snapshot.active.then_some(snapshot)
    }
    #[cfg(not(target_os = "android"))]
    {
        None
    }
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
        horizon_xr_boot::INPUT
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .menu_uv = None;
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
        // Diagnostic capture: every few seconds, save what the eyes
        // actually received (see `capture_xr`).
        if XR_TICK.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 300 == 0 {
            self.capture_xr(
                &[&xr_frame.eyes[0].texture, &xr_frame.eyes[1].texture],
                "xr-eyes.bmp",
            );
        }
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
        // The controller's aim ray against the screen: where it
        // lands, as UV over the menu texture, published for the
        // game's menu pointer (and used for the patch below, so the
        // player sees where the controller points). The right hand
        // aims when tracked, else the left.
        let aim_uv = {
            let snapshot = *horizon_xr_boot::INPUT
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let (aim, trigger) = match snapshot.aim_right {
                Some(aim) => (Some(aim), snapshot.right_trigger),
                None => (snapshot.aim_left, snapshot.left_trigger),
            };
            let uv = aim.and_then(|(quat, pos)| {
                let quat = glam::Quat::from_xyzw(quat[0], quat[1], quat[2], quat[3]);
                let dir = quat * glam::Vec3::NEG_Z;
                let origin = glam::Vec3::from_array(pos);
                let basis =
                    glam::Mat4::from_rotation_translation(screen.quat, screen.center);
                let normal = screen.quat * glam::Vec3::Z;
                let denom = dir.dot(normal);
                if denom.abs() <= 1e-6 {
                    return None;
                }
                let t = (screen.center - origin).dot(normal) / denom;
                if t <= 0.0 {
                    return None;
                }
                let hit = origin + dir * t;
                let local = basis.inverse() * hit.extend(1.0);
                let u = local.x / screen.width + 0.5;
                let v = 0.5 - local.y / screen.height;
                ((0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v)).then_some([u, v])
            });
            let mut guard = horizon_xr_boot::INPUT
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            guard.menu_uv = uv;
            guard.menu_trigger = trigger;
            uv
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
        // The pointer patch: where the aim ray crosses the screen
        // (or, with no controller tracked, the head's forward ray),
        // copied in as a small white square.
        let patch_uv = aim_uv.or_else(|| {
            let basis = glam::Mat4::from_rotation_translation(screen.quat, screen.center);
            let ray_dir = head_quat * glam::Vec3::NEG_Z;
            let normal = screen.quat * glam::Vec3::Z;
            let denom = ray_dir.dot(normal);
            if denom.abs() <= 1e-6 {
                return None;
            }
            let t = (screen.center - head_pos).dot(normal) / denom;
            if t <= 0.0 {
                return None;
            }
            let hit = head_pos + ray_dir * t;
            let local = basis.inverse() * hit.extend(1.0);
            let u = local.x / screen.width + 0.5;
            let v = 0.5 - local.y / screen.height;
            ((0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v)).then_some([u, v])
        });
        if let Some([u, v]) = patch_uv {
            let patch = 24u32;
            let x = ((u * self.config.width as f32) as i32 - patch as i32 / 2)
                .clamp(0, (self.config.width - patch) as i32) as u32;
            let y = ((v * self.config.height as f32) as i32 - patch as i32 / 2)
                .clamp(0, (self.config.height - patch) as i32) as u32;
            let white = white_patch(&self.device, &self.queue);
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("XR menu pointer patch"),
            });
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
        if XR_TICK.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 300 == 0 {
            self.capture_xr(&[&menu_texture], "xr-menu.bmp");
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

/// Frame counter for the diagnostic capture (see `capture_xr`).
#[cfg(target_os = "android")]
static XR_TICK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(target_os = "android")]
impl super::Renderer {
    /// Diagnostic capture (2026-11-23): read the given XR textures
    /// back and save them side by side, half size, as a BMP in the
    /// player's screenshots folder, so the exact images each eye
    /// receives can be inspected off-device. The file work runs on a
    /// detached thread; the render thread only pays the readback.
    fn capture_xr(&mut self, textures: &[&wgpu::Texture], name: &str) {
        // The textures' own size, not the renderer's config: the menu
        // texture is not the eye size by contract, only by today's
        // construction (agy review, 2026-11-23).
        let width = textures[0].width();
        let height = textures[0].height();
        let row = width * 4;
        let padded = row.div_ceil(256) * 256;
        let size = u64::from(padded) * u64::from(height);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("XR capture"),
        });
        let mut buffers = Vec::new();
        for texture in textures {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("XR capture"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded),
                        rows_per_image: Some(height),
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            buffers.push(buffer);
        }
        self.queue.submit([encoder.finish()]);
        let mut images = Vec::new();
        for buffer in &buffers {
            let slice = buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            match rx.recv() {
                Ok(Ok(())) => match slice.get_mapped_range() {
                    Ok(data) => {
                        images.push(data.to_vec());
                        drop(data);
                    }
                    Err(_) => {
                        log::warn!("XR_CAPTURE: readback failed");
                        return;
                    }
                },
                _ => {
                    log::warn!("XR_CAPTURE: readback failed");
                    return;
                }
            }
            buffer.unmap();
        }
        let path = crate::logging::player_dir().join("screenshots").join(name);
        std::thread::spawn(move || {
            // One writer at a time: two captures writing the same
            // file could interleave and corrupt it (agy review).
            static WRITER: std::sync::Mutex<()> = std::sync::Mutex::new(());
            let _guard = WRITER.lock();
            if write_side_by_side_bmp(&path, &images, width, height, padded).is_ok() {
                log::info!("XR_CAPTURE: saved {}", path.display());
            }
        });
    }
}

/// Write RGBA readbacks (rows padded to `padded` bytes) as one
/// 24-bit BMP: each image at half size, side by side.
#[cfg(target_os = "android")]
fn write_side_by_side_bmp(
    path: &std::path::Path,
    images: &[Vec<u8>],
    width: u32,
    height: u32,
    padded: u32,
) -> std::io::Result<()> {
    let half_w = width / 2;
    let half_h = height / 2;
    let out_w = half_w * images.len() as u32;
    let out_h = half_h;
    let row_bytes = (out_w * 3).div_ceil(4) * 4;
    let pixel_bytes = row_bytes * out_h;
    let file_size = 54 + pixel_bytes;
    let mut out = Vec::with_capacity(file_size as usize);
    // BITMAPFILEHEADER + BITMAPINFOHEADER.
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&file_size.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&54u32.to_le_bytes());
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(out_w as i32).to_le_bytes());
    out.extend_from_slice(&(out_h as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&pixel_bytes.to_le_bytes());
    out.extend_from_slice(&2835i32.to_le_bytes());
    out.extend_from_slice(&2835i32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    // Rows are stored bottom-up; each output pixel samples the
    // source at twice the coordinates (nearest neighbour, half size).
    for oy in 0..out_h {
        let sy = (out_h - 1 - oy) * 2;
        for image in images {
            for ox in 0..half_w {
                let sx = ox * 2;
                let at = (sy * padded + sx * 4) as usize;
                out.extend_from_slice(&[image[at + 2], image[at + 1], image[at]]);
            }
        }
        out.resize(out.len() + (row_bytes - out_w * 3) as usize, 0);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, out)
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
