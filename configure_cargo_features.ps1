<#
.SYNOPSIS
  Configure Cargo.toml for CPU (baseline) or CUDA (GPU) builds on Windows.

.DESCRIPTION
  Cargo.toml is committed with a CPU-only baseline (no cuda features). The Windows
  installer calls this script to layer CUDA features on top when an NVIDIA GPU +
  CUDA toolkit are present, mirroring what install.sh does on Linux. The edit is
  kept out of git via `git update-index --skip-worktree` so a GPU machine never
  commits CUDA features back for everyone.

  The script always normalizes to the CPU baseline first, then (for -Mode cuda)
  adds the features — so it is idempotent regardless of the file's current state.

.PARAMETER Mode
  cpu  -> strip all cuda features (CPU baseline)
  cuda -> CPU baseline + features = ["cuda"] on candle-core/nn/transformers + llama-cpp-2

.PARAMETER CargoPath
  Path to Cargo.toml. Defaults to zynkbot_rust\src-tauri\Cargo.toml next to this script.
#>
param(
    [ValidateSet('cpu','cuda')][string]$Mode = 'cpu',
    [string]$CargoPath
)

$ErrorActionPreference = 'Stop'
if (-not $CargoPath) {
    $here = Split-Path -Parent $MyInvocation.MyCommand.Path
    $CargoPath = Join-Path $here 'zynkbot_rust\src-tauri\Cargo.toml'
}
$rev = '1dd2d2c70a41d2969f13d5aa5c512251dc353773'

$lines = Get-Content $CargoPath

# 1) Normalize to CPU baseline (strip any cuda features)
$lines = $lines `
  -replace ('candle-core = \{ git = "https://github.com/huggingface/candle.git", rev = "'+$rev+'", features = \["cuda"\] \}'), ('candle-core = { git = "https://github.com/huggingface/candle.git", rev = "'+$rev+'" }') `
  -replace ('candle-nn = \{ git = "https://github.com/huggingface/candle.git", rev = "'+$rev+'", features = \["cuda"\] \}'), ('candle-nn = { git = "https://github.com/huggingface/candle.git", rev = "'+$rev+'" }') `
  -replace ('candle-transformers = \{ git = "https://github.com/huggingface/candle.git", rev = "'+$rev+'", features = \["cuda"\] \}'), ('candle-transformers = { git = "https://github.com/huggingface/candle.git", rev = "'+$rev+'" }') `
  -replace ('llama-cpp-2 = \{ version = "0.1", features = \["cuda"\] \}'), 'llama-cpp-2 = "0.1"'

# 2) If CUDA requested, add features onto the baseline
if ($Mode -eq 'cuda') {
    $lines = $lines `
      -replace ('candle-core = \{ git = "https://github.com/huggingface/candle.git", rev = "'+$rev+'" \}'), ('candle-core = { git = "https://github.com/huggingface/candle.git", rev = "'+$rev+'", features = ["cuda"] }') `
      -replace ('candle-nn = \{ git = "https://github.com/huggingface/candle.git", rev = "'+$rev+'" \}'), ('candle-nn = { git = "https://github.com/huggingface/candle.git", rev = "'+$rev+'", features = ["cuda"] }') `
      -replace ('candle-transformers = \{ git = "https://github.com/huggingface/candle.git", rev = "'+$rev+'" \}'), ('candle-transformers = { git = "https://github.com/huggingface/candle.git", rev = "'+$rev+'", features = ["cuda"] }') `
      -replace ('llama-cpp-2 = "0.1"'), 'llama-cpp-2 = { version = "0.1", features = ["cuda"] }'
}

Set-Content -Path $CargoPath -Value $lines
Write-Host "[configure_cargo_features] Cargo.toml set to '$Mode' baseline: $CargoPath"
