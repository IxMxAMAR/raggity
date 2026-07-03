$ErrorActionPreference = 'Stop'

$packageName = 'raggity'
$version     = '{{VERSION}}'
# SHA256 of raggity-<version>-windows-x86_64.zip from the GitHub release.
# Get it from the release page or: (Get-FileHash .\raggity-<ver>-windows-x86_64.zip -Algorithm SHA256).Hash
$checksum64  = '{{CHECKSUM}}'

$url64    = "https://github.com/IxMxAMAR/raggity/releases/download/v$version/raggity-$version-windows-x86_64.zip"
$toolsDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# The zip holds the onedir CONTENTS at its root (rag.exe alongside _internal\).
Install-ChocolateyZipPackage `
  -PackageName    $packageName `
  -Url64bit       $url64 `
  -Checksum64     $checksum64 `
  -ChecksumType64 'sha256' `
  -UnzipLocation  $toolsDir

# Expose `rag` on PATH. (Chocolatey also auto-shims *.exe in the package dir,
# but we register it explicitly so the command alias is deterministic.)
$exe = Join-Path $toolsDir 'rag.exe'
Install-BinFile -Name 'rag' -Path $exe
