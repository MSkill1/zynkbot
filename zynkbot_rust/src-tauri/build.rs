fn main() {
    println!("cargo:rerun-if-changed=seeds/einstein_seed.sql");

    // Desktop-only: point the linker at bundled libvosk.so for offline dictation.
    // libvosk.so from alphacep/vosk-api v0.3.45 (Linux x86_64)
    // sha256: 85c4654de3acdeb99abab86eeb2a6e603927d37089597c0fcc33d8638dc2ccaf
    #[cfg(not(target_os = "android"))]
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

    tauri_build::build()
}
