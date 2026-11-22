//! Stage 2 spike, step 2: the OpenXR loader smoke test.
//!
//! Once per process, at start-up (called from `crate::android::main` once the activity is
//! stored), this loads the bundled Khronos loader (`libopenxr_loader.so`, packaged by
//! build_apk.py), initialises it for this activity, creates an instance and asks it for the
//! head-mounted-display system — then logs the outcome and drops everything. No session, no
//! swapchain, no rendering: the step proves the loader and the device's runtime answer inside
//! the game's own process, and nothing more.
//!
//! Every line it logs carries the marker `XR_SMOKE`, under the game's logcat tag (`starling`,
//! android.rs's console sink). Failure is a warning and the game continues exactly as before:
//! a phone without an OpenXR runtime must notice nothing but the log line. Set
//! `HORIZON_NO_XR_SMOKE` in the environment (horizon-args.txt) to skip the probe entirely.

use std::sync::atomic::{AtomicBool, Ordering};

static RAN: AtomicBool = AtomicBool::new(false);

/// The probe, once per process. Never panics, never blocks start-up beyond the loader's own
/// work, and leaves no state behind (the instance is dropped before this returns).
pub fn run_once() {
    if RAN.swap(true, Ordering::SeqCst) {
        return;
    }
    if std::env::var_os("HORIZON_NO_XR_SMOKE").is_some() {
        return;
    }
    // The game's own logcat channel (android::logcat): the `log` crate's facade is not
    // bridged to it in this app, so `log::info!` would go nowhere (run #11 proved that —
    // the loader's own trace showed the probe working, but no marker line ever landed).
    match probe() {
        Ok((runtime, system)) => {
            crate::android::logcat(
                log::Level::Info,
                &format!("XR_SMOKE: ok runtime_name={runtime:?} system_name={system:?}"),
            );
        }
        Err(error) => {
            crate::android::logcat(
                log::Level::Warn,
                &format!("XR_SMOKE: unavailable: {error}"),
            );
        }
    }
}

fn probe() -> Result<(String, String), Box<dyn std::error::Error>> {
    let app = crate::android::android_app();
    // SAFETY: NativeActivity's VM and activity, alive as long as the process; the activity is
    // a global reference (the same handles android.rs's `with_activity` uses). The loader only
    // reads them while initialising and creating the instance, both inside this call.
    let platform_info = unsafe {
        openxr::AndroidPlatformInfo::new(app.vm_as_ptr(), app.activity_as_ptr())
    };
    // `loaded`: the loader is dlopen'd, not linked. On Android `Entry::load` resolves
    // "libopenxr_loader.so" from the app's library folder, where build_apk.py put it; every
    // Entry constructor runs the Android loader initialisation
    // (xrInitializeLoaderKHR with the VM and activity above) itself.
    // SAFETY: the packaged loader is the Khronos OpenXR loader, built to the specification.
    let entry = unsafe { openxr::Entry::load(&platform_info)? };
    let mut extensions = openxr::ExtensionSet::default();
    // Required to create an instance on Android at all; vulkan_enable2 is the session
    // extension step 3 will use, enabled now so its absence shows up in this step's log.
    extensions.khr_android_create_instance = true;
    extensions.khr_vulkan_enable2 = true;
    let instance = entry.create_instance(
        &openxr::ApplicationInfo {
            application_name: "Starling",
            application_version: 0,
            engine_name: "Starling XR smoke",
            engine_version: 0,
            api_version: openxr::Version::new(1, 0, 0),
        },
        &extensions,
        &[],
        &platform_info,
    )?;
    let runtime = instance.properties()?.runtime_name;
    let system = instance.system(openxr::FormFactor::HEAD_MOUNTED_DISPLAY)?;
    let system_name = instance.system_properties(system)?.system_name;
    // `instance` drops here: the probe keeps nothing.
    Ok((runtime, system_name))
}
