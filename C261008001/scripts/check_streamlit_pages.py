import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def main():
    results = []
    for path in [ROOT / "app.py", *sorted((ROOT / "pages").glob("*.py"))]:
        app = AppTest.from_file(str(path), default_timeout=45).run()
        errors = [str(item.value) for item in app.exception]
        results.append({"page": path.name, "passed": not errors, "errors": errors})
        print(path.name, "PASS" if not errors else errors)
    out = ROOT / "output" / "research_revision" / "page_execution_checks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
