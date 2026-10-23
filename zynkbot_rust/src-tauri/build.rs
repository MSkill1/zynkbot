fn main() {
    println!("cargo:rerun-if-changed=seeds/einstein_seed.sql");

    // Short git hash, stamped into "Report a problem" output. "unknown" when git
    // is absent (source tarball, CI without history).
    let git_hash = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=ZYNKBOT_GIT_HASH={}", git_hash);
    println!("cargo:rerun-if-changed=../.git/HEAD");

    // Android: the local-model engine is built with the OpenCL GPU backend (Cargo.toml,
    // feature `opencl`; 2026-10-23). llama-cpp-sys-2 compiles the backend but leaves the
    // final `-lOpenCL` to the top-level crate, so without this the app library carries
    // unresolved cl* symbols and Android refuses to load it. Link against the vendored
    // import library at build time; at run time the phone's own /vendor/lib64/libOpenCL.so
    // (a public library on Qualcomm and Pixel devices) is loaded by that name.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("android") {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        println!("cargo:rustc-link-search=native={}/android-opencl/lib", manifest_dir);
        println!("cargo:rustc-link-lib=dylib=OpenCL");
    }

    // Desktop-only: point the linker at bundled libvosk.so for offline dictation.
    // libvosk.so from alphacep/vosk-api v0.3.45 (Linux x86_64)
    // sha256: 85c4654de3acdeb99abab86eeb2a6e603927d37089597c0fcc33d8638dc2ccaf
    // libvosk.so is only available for Linux x86_64; gate all Vosk linker flags
    // to Linux so the Windows build doesn't try to find a non-existent libvosk.lib.
    #[cfg(target_os = "linux")]
    {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let vosk_lib_dir = format!("{}/lib/vosk", manifest_dir);
        println!("cargo:rustc-link-search=native={}", vosk_lib_dir);
        println!("cargo:rerun-if-changed={}/libvosk.so", vosk_lib_dir);
        // Embed rpath so the binary finds libvosk.so next to itself at runtime.
        // Three layouts covered:
        //   target/debug/app         → ../../lib/vosk  (cargo/tauri dev)
        //   /usr/bin/zynkbot         → ../lib/vosk     (typical .deb layout)
        //   /opt/zynkbot/zynkbot     → lib/vosk        (AppImage layout)
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../../lib/vosk");
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib/vosk");
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/lib/vosk");
    }

    // Windows: link against libvosk.lib from the Vosk Windows SDK, which install.bat
    // downloads into lib/vosk (not committed -- the SDK is ~66 MB). There is no rpath
    // on Windows; libvosk.dll and its three MinGW runtime DLLs are found at run time
    // via PATH, which START_ZYNKBOT.bat prepends when libvosk.dll is present.
    #[cfg(target_os = "windows")]
    {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let vosk_lib_dir = format!("{}/lib/vosk", manifest_dir);
        println!("cargo:rustc-link-search=native={}", vosk_lib_dir);
        println!("cargo:rerun-if-changed={}/libvosk.lib", vosk_lib_dir);
    }

    tauri_build::build()
}
