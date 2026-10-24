import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from esports_analysis.acceptance import generate_acceptance_markdown, run_acceptance_checks
from esports_analysis.config import get_database_path


def main() -> int:
    parser = argparse.ArgumentParser(description="运行项目自动验收并生成 Markdown 报告")
    parser.add_argument("--database", type=Path, default=get_database_path())
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / "system_acceptance_report.md",
    )
    args = parser.parse_args()

    checks, timings = run_acceptance_checks(args.database)
    report = generate_acceptance_markdown(checks, timings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")

    passed = sum(item.passed for item in checks)
    print(f"验收完成：{passed}/{len(checks)} 项通过")
    print(f"报告：{args.output.resolve()}")
    functional_failed = [
        item for item in checks
        if item.category != "论文研究数据" and not item.passed
    ]
    return 1 if functional_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

