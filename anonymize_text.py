from __future__ import annotations

import argparse
import base64
from pathlib import Path


def print_preserved_text(text: str) -> None:
    print(text, end="" if text.endswith("\n") else "\n")


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


def encode_normalized_text(normalized_text: str) -> str:
    return base64.b64encode(normalized_text.encode("utf-8")).decode("ascii")


def normalize_and_encode(text: str, print_normalized: bool = True) -> str:
    normalized = normalize_text(text)
    if print_normalized:
        print_preserved_text(normalized)
    return encode_normalized_text(normalized)


def anonymize_text(text: str, print_normalized: bool = True) -> str:
    return normalize_and_encode(text, print_normalized=print_normalized)


def anonymize_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    print_normalized: bool = True,
) -> str:
    encoded = anonymize_text(Path(input_path).read_text(), print_normalized=print_normalized)

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
