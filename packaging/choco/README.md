# Chocolatey package for raggity

Installs the standalone Windows `rag.exe` (no Python required) from the versioned
GitHub release zip.

## Files

- `raggity.nuspec` — package metadata (`{{VERSION}}` placeholder).
- `tools/chocolateyinstall.ps1` — downloads + unzips the release zip and shims
  `rag` onto PATH (`{{VERSION}}` and `{{CHECKSUM}}` placeholders).
- `tools/chocolateyuninstall.ps1` — removes the shim.

## Pack + push (per release)

Run from `packaging/choco/`. Replace the placeholders first — either edit the
files or use `sed`/`(Get-Content ...)`:

```powershell
$ver = '0.10.0'
# 1. Download the release zip and compute its checksum
$zip = "raggity-$ver-windows-x86_64.zip"
Invoke-WebRequest "https://github.com/IxMxAMAR/raggity/releases/download/v$ver/$zip" -OutFile $zip
$sum = (Get-FileHash $zip -Algorithm SHA256).Hash

# 2. Substitute placeholders (work on copies if you want to keep the templates)
(Get-Content raggity.nuspec)               -replace '{{VERSION}}',  $ver | Set-Content raggity.nuspec
(Get-Content tools/chocolateyinstall.ps1)  -replace '{{VERSION}}',  $ver -replace '{{CHECKSUM}}', $sum | Set-Content tools/chocolateyinstall.ps1

# 3. Pack + push to the Chocolatey community feed
choco pack raggity.nuspec
choco push raggity.$ver.nupkg --source https://push.chocolatey.org/ --api-key <YOUR_API_KEY>
```

The first community-repo submission goes through moderation/validation; automated
verification runs `choco install raggity` in a sandbox, so the release zip must be
public before pushing.
