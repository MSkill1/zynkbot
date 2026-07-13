# This script is no longer used.
#
# CUDA support is now implemented as a proper Cargo feature (see Cargo.toml [features]).
# START_ZYNKBOT.bat detects nvcc at launch time and passes --features cuda to tauri dev.
# Cargo.toml is never modified by the installer, so there is no skip-worktree issue.
Write-Host "[configure_cargo_features] No-op: CUDA is now handled via Cargo features at build time."
