$ErrorActionPreference = 'Stop'

# Remove the `rag` shim. The unpacked files under tools\ are removed by
# Chocolatey when the package is uninstalled.
Uninstall-BinFile -Name 'rag'
