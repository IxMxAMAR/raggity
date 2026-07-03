# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the raggity `rag` CLI (cross-platform, onedir).

OS-agnostic: builds a `rag` executable on Linux/macOS and `rag.exe` on Windows
(PyInstaller appends the platform suffix automatically). The onedir COLLECT
folder is named `rag`, so the default output is `dist/rag/rag[.exe]`.

Build (from repo root, with the venv active or via its python):
    pyinstaller packaging/raggity.spec --noconfirm --clean

Produces: dist/rag/rag[.exe]  (self-contained onedir bundle)

Spike/local build (isolated distpath):
    pyinstaller packaging/raggity.spec --noconfirm --clean \
        --distpath build-spike/dist --workpath build-spike/build

Key decisions (see packaging/README.md):
  * claude_agent_sdk ships a ~230MB bundled `claude.exe` under `_bundled/`.
    We EXCLUDE it: the SDK's _find_cli() falls back to `shutil.which("claude")`,
    so the Claude backend works as long as Claude Code is installed on PATH.
    Local backends (ollama/openai-compatible) need nothing extra.
  * Lazy/registry imports (store, embedder, reranker, answerer, connectors,
    claude_agent_sdk, llm_openai) are invisible to static analysis -> listed
    as hiddenimports below.
  * fastembed/onnxruntime/lancedb carry native DLLs + package data -> collect_all.
    fastembed downloads embedding models to the user cache at RUNTIME; none are
    bundled.
"""
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# --- repo layout ------------------------------------------------------------
# SPECPATH is packaging/; repo root is its parent.
REPO_ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))
SRC = os.path.join(REPO_ROOT, "src")
ENTRY = os.path.join(REPO_ROOT, "packaging", "rag_entry.py")

# --- collect native/data-heavy third-party packages -------------------------
datas = []
binaries = []
hiddenimports = []

for pkg in ("fastembed", "onnxruntime", "lancedb", "tokenizers", "pyarrow"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# rich/typer/click pull most things statically, but be safe on rich markup deps.
hiddenimports += collect_submodules("rich")

# --- raggity's own lazy imports (registry dotted-path + inline imports) ------
hiddenimports += [
    "raggity",
    "raggity.cli",
    "raggity.core",
    "raggity.store",           # resolve("store", "lancedb")
    "raggity.embedder",        # resolve("embedder", "fastembed")
    "raggity.reranker",        # resolve("reranker", "fastembed")
    "raggity.answerer",
    "raggity.cached_embedder",
    "raggity.llm",
    "raggity.llm_openai",      # ollama / openai backends
    "raggity.providers",
    "raggity.doctor",
    "raggity.conversation",
    "raggity.query_transform",
    "raggity.graph",
    "raggity.server",          # rag serve
    "raggity.connectors",
    "raggity.connectors.web",
    "raggity.connectors.github",
    "raggity.connectors.obsidian",
    # third-party lazy backends
    "claude_agent_sdk",
    "openai",
    "fastapi",
    "uvicorn",
]

# --- raggity data files (web UI served by `rag serve`) ----------------------
datas += [
    (os.path.join(SRC, "raggity", "web", "index.html"), os.path.join("raggity", "web")),
]

# --- keep the ~230MB bundled claude.exe OUT of the bundle -------------------
excludes = [
    # Heavy/unused; not raggity deps. Excluding trims size + avoids hook noise.
    # NOTE: PIL is required (fastembed imports it for image-embedding support).
    "tkinter", "matplotlib", "PyQt5", "PySide2", "notebook", "IPython",
    "pytest",
    # OCR stack: raggity[ocr] optional extra only (readers.py imports these
    # lazily inside functions). Excluding drops ~110MB (cv2 is ~99MB). A base
    # binary does not ship OCR; users who need scanned-PDF/image OCR install
    # raggity[ocr] into a Python env. Remove these 3 to build an OCR-capable exe.
    "cv2", "rapidocr_onnxruntime", "pypdfium2",
]

a = Analysis(
    [ENTRY],
    pathex=[SRC],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Drop the SDK's bundled ~230MB claude CLI from the collected tree. The SDK
# falls back to a system `claude` on PATH when the bundled one is absent.
# NOTE: match the `_bundled` path segment (not a fixed `claude.exe` filename):
# on Linux/macOS the bundled binary is `_bundled/claude` with no `.exe` suffix,
# so an endswith("claude.exe") filter would silently miss it there.
def _is_sdk_bundled(path):
    p = path.replace("\\", "/")
    return "claude_agent_sdk" in p and "_bundled" in p

a.datas = [d for d in a.datas if not _is_sdk_bundled(d[0])]
a.binaries = [b for b in a.binaries if not _is_sdk_bundled(b[0])]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="rag",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="rag",
)
