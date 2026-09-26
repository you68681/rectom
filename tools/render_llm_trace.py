from __future__ import annotations

import argparse
import json
from pathlib import Path


def indented_block(value: object) -> str:
    text = "" if value is None else str(value)
    if not text:
        text = "(empty)"
    return "\n".join(f"    {line}" for line in text.splitlines())


def render_trace(source: Path, destination: Path) -> None:
    records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output = [
        f"# LLM trace: {source.parent.name}",
        "",
        f"Source: `{source}`",
        "",
        f"Total calls: {len(records)}",
        "",
    ]
    for index, record in enumerate(records, start=1):
        output.extend(
            [
                f"## Call {index}: {record.get('call_type', 'unknown')}",
                "",
                f"- Provider: `{record.get('provider')}`",
                f"- Model: `{record.get('model')}`",
                f"- Attempt: `{record.get('attempt')}`",
                "- Generation parameters:",
                "",
                indented_block(
                    json.dumps(
                        record.get("generation_parameters", {}),
                        ensure_ascii=False,
                        indent=2,
                    )
                ),
                "",
                "### Prompt",
                "",
                indented_block(record.get("prompt")),
                "",
                "### Response",
                "",
                indented_block(record.get("response")),
                "",
                "### Error",
                "",
                indented_block(record.get("error")),
                "",
            ]
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a full LLM JSONL trace as readable Markdown."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    render_trace(args.source, args.destination)


if __name__ == "__main__":
    main()
