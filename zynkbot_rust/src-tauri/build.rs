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

    // Android: the local-model engine is built as shared libraries with loadable backends
    // (Cargo.toml: llama-cpp-2 feature `dynamic-backends`, 2026-10-23; the OpenCL backend
    // was dropped on 2026-10-24 after measuring it). The engine crate's CMake build leaves
    // libllama/libggml*.so and the CPU-variant modules in its own output folder; the APK
    // only packages what sits in jniLibs, so copy them there. The crate's output folder is
    // a sibling of ours under target/<triple>/<profile>/build/.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("android") {
        let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
        let build_dir = out_dir.ancestors().nth(2).expect("OUT_DIR under target/.../build/<pkg>/out");
        let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
        let jni = manifest_dir.join("gen/android/app/src/main/jniLibs/arm64-v8a");
        let _ = std::fs::create_dir_all(&jni);
        let mut newest: Option<(std::time::SystemTime, std::path::PathBuf)> = None;
        if let Ok(entries) = std::fs::read_dir(build_dir) {
            for e in entries.flatten() {
                let p = e.path();
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if !name.starts_with("llama-cpp-sys-2-") { continue; }
                let out = p.join("out");
                if !out.join("backends").is_dir() { continue; }
                let t = std::fs::metadata(&out).and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
                if newest.as_ref().map_or(true, |(nt, _)| t > *nt) { newest = Some((t, out)); }
            }
        }
        // Backend module names are baked into the app (ZB_ANDROID_BACKENDS) because the APK
        // keeps its libraries inside base.apk: there is no folder to scan at run time, so
        // local_models.rs loads them by name.
        let mut module_names: Vec<String> = Vec::new();
        if let Some((_, out)) = newest {
            // Drop engine files from an earlier build first; a module that no longer exists
            // must not ride along, and libllama-common.so (65 MB, nothing links it) never does.
            if let Ok(entries) = std::fs::read_dir(&jni) {
                for e in entries.flatten() {
                    let name = e.file_name().to_string_lossy().into_owned();
                    if (name.starts_with("libggml") || name.starts_with("libllama")) && e.file_type().map_or(false, |t| t.is_file()) {
                        let _ = std::fs::remove_file(e.path());
                    }
                }
            }
            let mut copied = 0;
            for sub in ["lib", "backends", "lib64", "bin"] {
                if let Ok(entries) = std::fs::read_dir(out.join(sub)) {
                    for e in entries.flatten() {
                        let p = e.path();
                        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
                        if !name.ends_with(".so") || name == "libllama-common.so" { continue; }
                        if std::fs::copy(&p, jni.join(&name)).is_ok() {
                            copied += 1;
                            if sub == "backends" { module_names.push(name); }
                        }
                    }
                }
            }
            module_names.sort();
            println!("cargo:warning=android: copied {} engine .so file(s) from {} into jniLibs; modules: {}", copied, out.display(), module_names.join(","));
        } else {
            println!("cargo:warning=android: no llama-cpp-sys-2 output with a backends/ folder found under {}", build_dir.display());
        }
        println!("cargo:rustc-env=ZB_ANDROID_BACKENDS={}", module_names.join(","));
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
