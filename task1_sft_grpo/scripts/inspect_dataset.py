"""Dataset inspection script."""

import argparse
import json
import os

from sft_lab.dataset import inspect_dataset_file


def format_markdown_report(report: dict) -> str:
    """Format json report into human-readable Markdown report."""
    md = []
    md.append("# Dataset Inspection Report\n")
    md.append(f"- **Dataset File**: `{report['dataset_file']}`")
    md.append(f"- **Tokenizer**: `{report['tokenizer_name']}`")
    md.append(f"- **Max Sequence Length**: `{report['max_seq_length']}`")
    md.append(f"- **Total Examples**: `{report['total_examples']}`")
    md.append(f"- **Validation Issues Count**: `{report['validation_issues_count']}`\n")

    md.append("## Structure & Validation Issues\n")
    if report["validation_issues_count"] == 0:
        md.append("No structural validation issues detected.\n")
    else:
        for issue_type, msgs in report["issues_by_type"].items():
            md.append(f"### `{issue_type}` ({len(msgs)} occurrences)")
            for msg in msgs[:10]:  # Show top 10
                md.append(f"- {msg}")
            if len(msgs) > 10:
                md.append(f"- ... and {len(msgs) - 10} more")
            md.append("")

    md.append("## Duplicates\n")
    dup = report["duplicates"]
    md.append(f"- **Duplicate IDs**: {dup['duplicate_ids_count']}")
    md.append(f"- **Exact Duplicate Conversations**: {dup['exact_duplicate_conversations']}")
    md.append(f"- **Exact Duplicate Rendered Sequences**: {dup['exact_duplicate_rendered']}\n")

    md.append("## Summary Statistics\n")
    md.append("| Metric | Min | Mean | Median | P90 | P95 | P99 | Max |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for stat_name, s in report["stats"].items():
        md.append(
            f"| **{stat_name}** | {s['min']:.1f} | {s['mean']:.1f} | {s['median']:.1f} | {s['p90']:.1f} | {s['p95']:.1f} | {s['p99']:.1f} | {s['max']:.1f} |"
        )
    md.append("")

    md.append("## Truncation Analysis\n")
    tr = report["truncation"]
    md.append(f"- **Truncated Examples**: {tr['truncated_examples']} ({tr['truncated_percentage']:.2f}%)")
    md.append(
        f"- **Assistant Truncated Examples**: {tr['assistant_truncated_examples']} ({tr['assistant_truncated_percentage']:.2f}%)"
    )
    md.append(f"- **Supervised Tokens Lost**: {tr['supervised_tokens_lost']}")
    md.append(f"- **Zero Supervised Token Examples**: {tr['zero_supervised_examples']}\n")

    md.append("## Sources & Categories\n")
    md.append("### Sources")
    for src, data in report["sources"].items():
        md.append(f"- **{src}**: {data['count']} examples, {data['tokens']} tokens")
    md.append("\n### Categories")
    for cat, data in report["categories"].items():
        md.append(f"- **{cat}**: {data['count']} examples, {data['tokens']} tokens")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Inspect dataset structure, tokens, and quality.")
    parser.add_argument("--data_path", type=str, default="data/processed/sft_data.jsonl", help="Dataset path")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Tokenizer ID")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")
    parser.add_argument(
        "--json_output",
        type=str,
        default="artifacts/reports/dataset_report.json",
        help="JSON report path",
    )
    parser.add_argument(
        "--md_output",
        type=str,
        default="artifacts/reports/dataset_report.md",
        help="Markdown report path",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.json_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.md_output), exist_ok=True)

    print(f"Inspecting dataset: {args.data_path} with tokenizer {args.tokenizer_name}...")
    report = inspect_dataset_file(args.data_path, args.tokenizer_name, args.max_seq_length)

    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_content = format_markdown_report(report)
    with open(args.md_output, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Report saved to:\n  JSON: {args.json_output}\n  Markdown: {args.md_output}")


if __name__ == "__main__":
    main()
