from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIBREOFFICE_BIN = ROOT / "tools" / "LibreOffice" / "program"
POPPLER_BIN = Path(
    r"C:\Users\29148\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin"
)
RENDERER = Path(
    r"C:\Users\29148\.codex\plugins\cache\openai-primary-runtime\documents\26.1011.11113\skills\documents\render_docx.py"
)


def main() -> None:
    soffice = LIBREOFFICE_BIN / "soffice.exe"
    if not soffice.exists():
        raise FileNotFoundError(f"未找到项目内LibreOffice: {soffice}")
    if not RENDERER.exists():
        raise FileNotFoundError(f"未找到文档渲染器: {RENDERER}")
    current_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(
        [str(LIBREOFFICE_BIN), str(POPPLER_BIN), current_path]
    )
    sys.argv[0] = str(RENDERER)
    runpy.run_path(str(RENDERER), run_name="__main__")


if __name__ == "__main__":
    main()
