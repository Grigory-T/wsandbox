from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HEADER_PREFIX = "\t Вид"
DATA_ROW_PREFIX = "РМ"
ANONYMIZED_DATA_ROW_PREFIX = "AA"
SPECIAL_SEPARATOR_INDEX = 2
TABLE_GAP = 4
MIN_HEADER_TABS = 2
TAB_RUN_RE = re.compile(r"\t+")


@dataclass
class PhysicalRow:
    line_number: int
    raw: str


@dataclass
class ParsedRow:
    line_numbers: list[int]
    raw: str
    cells: list[str]
    issues: list[str]

    @property
    def line_number(self) -> int:
        return self.line_numbers[0]


@dataclass
class TableDebug:
    physical_rows: int = 0
    empty_header_columns: int = 0
    blank_rows_removed: int = 0
    data_rows_started: int = 0
    continuation_rows_joined: int = 0
    orphan_continuation_rows: int = 0
    rows_with_short_right_side: int = 0
    rows_with_extra_columns: int = 0
    rows_with_delimiter_issues: int = 0
    rows_using_special_separator_width: int = 0


@dataclass
class ParsedTable:
    header_line_number: int
    header_raw: str
    header: list[str]
    separator_widths: list[int]
    rows: list[ParsedRow]
    debug: TableDebug


@dataclass
class ParseDebug:
    physical_rows: int = 0
    skipped_leading_rows: int = 0
    headers_found: int = 0


@dataclass
class ParseResult:
    tables: list[ParsedTable]
    debug: ParseDebug


def read_table_text(path: Path) -> str:
    text = path.read_text()
    if "\n" not in text and ("\\n" in text or "\\t" in text):
        text = text.replace("\\n", "\n").replace("\\t", "\t")
    return text


def split_physical_rows(text: str) -> list[PhysicalRow]:
    return [
        PhysicalRow(line_number=line_number, raw=raw)
        for line_number, raw in enumerate(text.splitlines(), start=1)
    ]


def is_header_row(row: PhysicalRow) -> bool:
    return row.raw.startswith(HEADER_PREFIX)


def is_data_row(raw: str, allow_anonymized_prefix: bool = False) -> bool:
    text = raw.lstrip("\t ")
    return text.startswith(DATA_ROW_PREFIX) or (
        allow_anonymized_prefix and text.startswith(ANONYMIZED_DATA_ROW_PREFIX)
    )


def is_fallback_header_row(row: PhysicalRow) -> bool:
    return row.raw.count("\t") >= MIN_HEADER_TABS and not is_data_row(
        row.raw,
        allow_anonymized_prefix=True,
    )


def header_parse_text(raw: str) -> str:
    if raw.startswith("\t"):
        return raw[1:]
    return raw


def parse_header(raw: str) -> tuple[list[str], list[int]]:
    text = header_parse_text(raw)
    columns = [part.strip() for part in TAB_RUN_RE.split(text)]
    separator_widths = [len(match.group(0)) for match in TAB_RUN_RE.finditer(text)]
    return columns, separator_widths


def split_tables(
    rows: list[PhysicalRow],
    debug: ParseDebug,
) -> list[tuple[PhysicalRow, list[PhysicalRow]]]:
    tables: list[tuple[PhysicalRow, list[PhysicalRow]]] = []
    current: tuple[PhysicalRow, list[PhysicalRow]] | None = None
    blank_run = 0

    for row in rows:
        if not row.raw.strip():
            blank_run += 1
            if current is None:
                debug.skipped_leading_rows += 1
            else:
                current[1].append(row)
            continue

        starts_table_by_marker = is_header_row(row)
        starts_first_table_by_shape = current is None and is_fallback_header_row(row)
        starts_next_table_by_shape = (
            current is not None
            and blank_run >= TABLE_GAP
            and is_fallback_header_row(row)
        )

        if starts_table_by_marker or starts_first_table_by_shape or starts_next_table_by_shape:
            current = (row, [])
            tables.append(current)
            debug.headers_found += 1
            blank_run = 0
            continue

        if current is None:
            debug.skipped_leading_rows += 1
            blank_run = 0
            continue

        current[1].append(row)
        blank_run = 0

    return tables


def separator_width_options(index: int, expected_width: int) -> tuple[int, ...]:
    if index == SPECIAL_SEPARATOR_INDEX:
        return tuple(sorted({expected_width, 1, 2}))
    return (expected_width,)


def resolve_separator_count(
    separator_widths: list[int],
    start_index: int,
    run_width: int,
) -> int | None:
    sums = {0}
    for end_index in range(start_index, len(separator_widths)):
        sums = {
            total + option
            for total in sums
            for option in separator_width_options(end_index, separator_widths[end_index])
        }
        if run_width in sums:
            return end_index - start_index + 1
        sums = {total for total in sums if total <= run_width}
        if not sums:
            return None
    return None


def fallback_separator_count(
    separator_widths: list[int],
    start_index: int,
    run_width: int,
) -> int:
    if start_index >= len(separator_widths):
        return 1

    total = 0
    consumed = 0
    for index in range(start_index, len(separator_widths)):
        total += min(separator_width_options(index, separator_widths[index]))
        consumed += 1
        if total >= run_width:
            return consumed
    return max(consumed, 1)


def tokenize_tab_runs(raw: str) -> tuple[list[str], list[int]]:
    values: list[str] = []
    run_widths: list[int] = []
    previous_end = 0

    for match in TAB_RUN_RE.finditer(raw):
        values.append(raw[previous_end : match.start()])
        run_widths.append(len(match.group(0)))
        previous_end = match.end()

    values.append(raw[previous_end:])
    return values, run_widths


def parse_cells(
    raw: str,
    header_width: int,
    separator_widths: list[int],
    drop_leading_tab: bool = False,
) -> tuple[list[str], list[str], bool, bool]:
    if drop_leading_tab and raw.startswith("\t"):
        raw = raw[1:]

    values, run_widths = tokenize_tab_runs(raw)
    cells = [values[0]]
    separator_index = 0
    issues: list[str] = []
    used_special_separator_width = False

    for run_width, next_value in zip(run_widths, values[1:]):
        if separator_index >= len(separator_widths):
            cells.append(next_value)
            issues.append(f"extra delimiter run of {run_width} tabs after header columns")
            continue

        separator_count = resolve_separator_count(
            separator_widths,
            separator_index,
            run_width,
        )
        if separator_count is None:
            separator_count = fallback_separator_count(
                separator_widths,
                separator_index,
                run_width,
            )
            issues.append(
                f"delimiter run of {run_width} tabs does not match header layout "
                f"at separator {separator_index + 1}"
            )

        if (
            separator_index == SPECIAL_SEPARATOR_INDEX
            and separator_count == 1
            and run_width in {1, 2}
            and run_width != separator_widths[separator_index]
        ):
            used_special_separator_width = True

        cells.extend([""] * (separator_count - 1))
        cells.append(next_value)
        separator_index += separator_count

    short_right_side = len(cells) < header_width
    if short_right_side:
        cells.extend([""] * (header_width - len(cells)))

    if len(cells) > header_width:
        issues.append(f"parsed {len(cells)} cells for {header_width} header columns")

    return cells, issues, short_right_side, used_special_separator_width


def join_data_rows(
    rows: list[PhysicalRow],
    debug: TableDebug,
    allow_anonymized_prefix: bool = False,
) -> list[tuple[list[int], str, list[str]]]:
    joined_rows: list[tuple[list[int], str, list[str]]] = []

    for row in rows:
        if not row.raw.strip():
            debug.blank_rows_removed += 1
            continue

        if is_data_row(row.raw, allow_anonymized_prefix=allow_anonymized_prefix):
            joined_rows.append(([row.line_number], row.raw, []))
            debug.data_rows_started += 1
            continue

        if not joined_rows:
            joined_rows.append(
                (
                    [row.line_number],
                    row.raw,
                    ["non-RM row has no previous data row to join"],
                )
            )
            debug.orphan_continuation_rows += 1
            continue

        debug.continuation_rows_joined += 1
        line_numbers, raw, issues = joined_rows[-1]
        line_numbers.append(row.line_number)
        joined_rows[-1] = (line_numbers, raw + row.raw, issues)

    return joined_rows


def parse_document(text: str) -> ParseResult:
    rows = split_physical_rows(text)
    debug = ParseDebug(physical_rows=len(rows))
    tables: list[ParsedTable] = []

    for header_row, body_rows in split_tables(rows, debug):
        header, separator_widths = parse_header(header_row.raw)
        drop_leading_tab = header_row.raw.startswith("\t")
        allow_anonymized_prefix = not is_header_row(header_row)
        table_debug = TableDebug(physical_rows=1 + len(body_rows))
        table_debug.empty_header_columns = sum(1 for column in header if not column)
        parsed_rows: list[ParsedRow] = []

        for line_numbers, raw, row_issues in join_data_rows(
            body_rows,
            table_debug,
            allow_anonymized_prefix=allow_anonymized_prefix,
        ):
            cells, cell_issues, short_right_side, used_special_separator_width = parse_cells(
                raw,
                len(header),
                separator_widths,
                drop_leading_tab=drop_leading_tab,
            )
            issues = row_issues + cell_issues
            if short_right_side:
                table_debug.rows_with_short_right_side += 1
            if len(cells) > len(header):
                table_debug.rows_with_extra_columns += 1
            if cell_issues:
                table_debug.rows_with_delimiter_issues += 1
            if used_special_separator_width:
                table_debug.rows_using_special_separator_width += 1

            parsed_rows.append(
                ParsedRow(
                    line_numbers=line_numbers,
                    raw=raw,
                    cells=cells,
                    issues=issues,
                )
            )

        tables.append(
            ParsedTable(
                header_line_number=header_row.line_number,
                header_raw=header_row.raw,
                header=header,
                separator_widths=separator_widths,
                rows=parsed_rows,
                debug=table_debug,
            )
        )

    return ParseResult(tables=tables, debug=debug)


def parse_tables(text: str) -> list[ParsedTable]:
    return parse_document(text).tables


def parse_tables_to_dataframes(path: str | Path) -> list[Any]:
    import pandas as pd

    result = parse_document(read_table_text(Path(path)))
    dataframes = []

    for table_number, table in enumerate(result.tables, start=1):
        meta_columns = ["table_number", "row_number", "physical_row_numbers"]
        columns = meta_columns + table.header
        records = []

        for row_number, row in enumerate(table.rows, start=1):
            records.append(
                [table_number, row_number, row.line_numbers] + row.cells[: len(table.header)]
            )

        dataframes.append(pd.DataFrame(records, columns=columns))

    return dataframes


def parse_file(path: str | Path) -> list[Any]:
    return parse_tables_to_dataframes(path)


def summarize_result(result: ParseResult) -> dict[str, object]:
    return {
        "debug": {
            "physical_rows": result.debug.physical_rows,
            "skipped_leading_rows": result.debug.skipped_leading_rows,
            "headers_found": result.debug.headers_found,
        },
        "tables": [
            {
                "table_number": index,
                "header_line": table.header_line_number,
                "header_columns": len(table.header),
                "rows": len(table.rows),
                "row_columns_min": min(
                    (len(row.cells) for row in table.rows),
                    default=0,
                ),
                "row_columns_max": max(
                    (len(row.cells) for row in table.rows),
                    default=0,
                ),
                "separator_widths": table.separator_widths,
                "debug": {
                    "physical_rows": table.debug.physical_rows,
                    "empty_header_columns": table.debug.empty_header_columns,
                    "blank_rows_removed": table.debug.blank_rows_removed,
                    "data_rows_started": table.debug.data_rows_started,
                    "continuation_rows_joined": table.debug.continuation_rows_joined,
                    "orphan_continuation_rows": table.debug.orphan_continuation_rows,
                    "rows_with_short_right_side": table.debug.rows_with_short_right_side,
                    "rows_with_extra_columns": table.debug.rows_with_extra_columns,
                    "rows_with_delimiter_issues": table.debug.rows_with_delimiter_issues,
                    "rows_using_special_separator_width": (
                        table.debug.rows_using_special_separator_width
                    ),
                },
                "row_issues": [
                    {
                        "line_numbers": row.line_numbers,
                        "issues": row.issues,
                    }
                    for row in table.rows
                    if row.issues
                ],
            }
            for index, table in enumerate(result.tables, start=1)
        ]
    }


def summarize_tables(tables: list[ParsedTable]) -> dict[str, object]:
    return summarize_result(ParseResult(tables=tables, debug=ParseDebug()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    result = parse_document(read_table_text(args.path))
    print(json.dumps(summarize_result(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
