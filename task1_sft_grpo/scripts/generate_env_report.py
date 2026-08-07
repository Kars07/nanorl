"""Generate environment.json report."""

import json
import os

from sft_lab.seed import generate_environment_report


def main():
    os.makedirs("artifacts/reports", exist_ok=True)
    report = generate_environment_report()
    out_path = "artifacts/reports/environment.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Environment report saved to {out_path}")


if __name__ == "__main__":
    main()
