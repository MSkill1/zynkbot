$cargoPath = "$env:USERPROFILE\.cargo\bin"
$machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
if ($machinePath -notlike "*$cargoPath*") {
    [Environment]::SetEnvironmentVariable('Path', "$cargoPath;$machinePath", 'Machine')
    Write-Host "Added Rust to system PATH"
} else {
    Write-Host "Rust already in system PATH"
}
