from __future__ import annotations

import argparse
import base64
from pathlib import Path


def normalize_text(text: str) -> str:
    result = []

    for ch in text:
        if ch.isalpha():
            result.append("A")
        elif ch.isdigit():
            result.append("1")
        else:
            result.append(ch)

    return "".join(result)


def normalize_and_encode(text: str) -> str:
    normalized = normalize_text(text)
    return base64.b64encode(normalized.encode("utf-8")).decode("ascii")


def anonymize_file(input_path: str | Path, output_path: str | Path | None = None) -> str:
    encoded = normalize_and_encode(Path(input_path).read_text())

    if output_path is not None:
        Path(output_path).write_text(encoded)

    return encoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("-o", "--output-path", type=Path)
    args = parser.parse_args()

    encoded = anonymize_file(args.input_path, args.output_path)
    if args.output_path is None:
        print(encoded)


if __name__ == "__main__":
    main()
