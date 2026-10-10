# PyInstaller recipe for the Aria app. From the repository root:
#
#     pip install pyinstaller pillow
#     pyinstaller --noconfirm packaging/aria.spec
#
# makes dist/Aria.app on a Mac, and dist/Aria/ (with Aria.exe or Aria in it)
# on Windows and Linux. Each is built on its own system: a Mac can't build the
# Windows app or the other way round. .github/workflows/apps.yml builds all
# three, checks each with `--self-test`, and packages them.

import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).parent
ICON = ROOT / "aria" / "icon.png"      # converted to .icns / .ico (needs Pillow)
VERSION = re.search(r'__version__ = "([^"]+)"',
                    (ROOT / "aria" / "__init__.py").read_text(encoding="utf-8")).group(1)

a = Analysis(
    [str(ROOT / "packaging" / "aria_app.py")],
    pathex=[str(ROOT)],
    datas=[
        (str(ROOT / "checkpoints" / "aria-small.pt"), "checkpoints"),
        (str(ROOT / "aria" / "seed_dialogues.txt"), "aria"),
        (str(ICON), "aria"),
    ],
    hiddenimports=collect_submodules("aria"),
    # Things that may be installed on the build machine but Aria never uses.
    excludes=["PIL", "matplotlib", "IPython", "jupyter", "notebook", "pandas",
              "scipy", "pytest", "torchvision", "torchaudio", "tensorboard"],
    # Parts of PyTorch read their own source code at import time.
    module_collection_mode={"torch": "pyz+py"},
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Aria",
    console=False,               # a window of its own, not a terminal
    icon=str(ICON),
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="Aria", upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Aria.app",
        icon=str(ICON),
        bundle_identifier="io.github.rocket-ramuel.aria",
        version=VERSION,
        info_plist={
            "CFBundleDisplayName": "Aria",
            "CFBundleShortVersionString": VERSION,
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
        },
    )
