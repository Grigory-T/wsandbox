from __future__ import annotations

import argparse
import base64
import binascii
import json
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HEADER_PREFIX = "\t Вид"
DATA_ROW_PREFIX = "PM"
ANONYMIZED_DATA_ROW_PREFIX = "AA"
SPECIAL_SEPARATOR_INDEX = 2
TABLE_GAP = 2
MIN_HEADER_TABS = 2
TAB_RUN_RE = re.compile(r"\t+")
DEFAULT_TEXT_ENCODINGS = ("utf-8-sig", "cp1251", "cp1252")
ROW_START_IGNORED_CHARS = "\ufeff\u200b\u200c\u200d\xa0 \t"


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


@dataclass
class FolderParseResult:
    dataframes: list[Any]
    errors: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    files_total: int
    files_ok: int
    files_failed: int
    tables_total: int


def read_text_file(path: str | Path, encoding: str | None = None) -> str:
    path = Path(path)
    if encoding is not None:
        return path.read_text(encoding=encoding)

    last_error: UnicodeDecodeError | None = None
    for candidate_encoding in DEFAULT_TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=candidate_encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    assert last_error is not None
    last_error.add_note(f"Tried encodings: {', '.join(DEFAULT_TEXT_ENCODINGS)}")
    raise last_error


def read_table_text(path: str | Path, encoding: str | None = None) -> str:
    text = read_text_file(path, encoding=encoding)
    text = decode_base64_text_if_needed(text)
    if "\n" not in text and ("\\n" in text or "\\t" in text):
        text = text.replace("\\n", "\n").replace("\\t", "\t")
    return text


def decode_base64_text_if_needed(text: str) -> str:
    if "\t" in text or "\\t" in text:
        return text

    compact = "".join(text.split())
    if not compact:
        return text

    try:
        decoded_bytes = base64.b64decode(compact, validate=True)
        decoded = decoded_bytes.decode("utf-8-sig")
    except (binascii.Error, UnicodeDecodeError):
        return text

    if "\t" in decoded or "\\t" in decoded:
        return decoded

    return text


def split_physical_rows(text: str) -> list[PhysicalRow]:
    return [
        PhysicalRow(line_number=line_number, raw=raw)
        for line_number, raw in enumerate(text.splitlines(), start=1)
    ]


def is_header_row(row: PhysicalRow) -> bool:
    return row.raw.lstrip("\ufeff\u200b\u200c\u200d").startswith(HEADER_PREFIX)


def first_cell_for_classification(raw: str, drop_leading_tab: bool = False) -> str:
    if drop_leading_tab and raw.startswith("\t"):
        raw = raw[1:]

    text = raw.lstrip(ROW_START_IGNORED_CHARS)
    return text.split("\t", 1)[0].strip(ROW_START_IGNORED_CHARS)


def is_data_row(
    raw: str,
    allow_anonymized_prefix: bool = False,
    drop_leading_tab: bool = False,
) -> bool:
    first_cell = first_cell_for_classification(raw, drop_leading_tab=drop_leading_tab)

    if first_cell.upper().startswith(DATA_ROW_PREFIX):
        return True

    if not allow_anonymized_prefix:
        return False

    return (
        first_cell.startswith(ANONYMIZED_DATA_ROW_PREFIX)
        and any(ch.isdigit() for ch in first_cell)
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
    drop_leading_tab: bool = False,
) -> list[tuple[list[int], str, list[str]]]:
    joined_rows: list[tuple[list[int], str, list[str]]] = []

    for row in rows:
        if not row.raw.strip():
            debug.blank_rows_removed += 1
            continue

        if is_data_row(
            row.raw,
            allow_anonymized_prefix=allow_anonymized_prefix,
            drop_leading_tab=drop_leading_tab,
        ):
            joined_rows.append(([row.line_number], row.raw, []))
            debug.data_rows_started += 1
            continue

        if not joined_rows:
            joined_rows.append(
                (
                    [row.line_number],
                    row.raw,
                    ["non-PM row has no previous data row to join"],
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
            drop_leading_tab=drop_leading_tab,
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


def parsed_result_to_dataframes(
    result: ParseResult,
    source_path: str | Path | None = None,
    file_number: int | None = None,
) -> list[Any]:
    import pandas as pd

    dataframes = []

    for table_number, table in enumerate(result.tables, start=1):
        meta_columns = []
        meta_values: list[Any] = []

        if file_number is not None:
            meta_columns.append("file_number")
            meta_values.append(file_number)

        if source_path is not None:
            source = Path(source_path)
            meta_columns.extend(["source_file", "source_path"])
            meta_values.extend([source.name, str(source)])

        meta_columns.extend(["table_number", "row_number", "physical_row_numbers"])
        columns = meta_columns + table.header
        records = []

        for row_number, row in enumerate(table.rows, start=1):
            records.append(
                meta_values
                + [table_number, row_number, row.line_numbers]
                + row.cells[: len(table.header)]
            )

        dataframes.append(pd.DataFrame(records, columns=columns))

    return dataframes


def parse_tables_to_dataframes(path: str | Path, encoding: str | None = None) -> list[Any]:
    return parsed_result_to_dataframes(parse_document(read_table_text(path, encoding=encoding)))


def parse_file(path: str | Path, encoding: str | None = None) -> list[Any]:
    return parse_tables_to_dataframes(path, encoding=encoding)


def iter_txt_files(folder_path: str | Path, recursive: bool = False) -> list[Path]:
    folder = Path(folder_path)
    pattern = "**/*.txt" if recursive else "*.txt"
    return sorted(path for path in folder.glob(pattern) if path.is_file())


def table_warning_records(
    table: ParsedTable,
    file_number: int,
    source_path: Path,
    table_number: int,
) -> list[dict[str, Any]]:
    warnings = []
    row_issues = [
        {"line_numbers": row.line_numbers, "issues": row.issues}
        for row in table.rows
        if row.issues
    ]

    if row_issues:
        warnings.append(
            {
                "file_number": file_number,
                "source_file": source_path.name,
                "source_path": str(source_path),
                "table_number": table_number,
                "kind": "row_issues",
                "details": row_issues,
            }
        )

    debug_counts = {
        "continuation_rows_joined": table.debug.continuation_rows_joined,
        "orphan_continuation_rows": table.debug.orphan_continuation_rows,
        "rows_with_extra_columns": table.debug.rows_with_extra_columns,
        "rows_with_delimiter_issues": table.debug.rows_with_delimiter_issues,
    }
    nonzero_debug_counts = {key: value for key, value in debug_counts.items() if value}

    if nonzero_debug_counts:
        warnings.append(
            {
                "file_number": file_number,
                "source_file": source_path.name,
                "source_path": str(source_path),
                "table_number": table_number,
                "kind": "parse_debug_counts",
                "details": nonzero_debug_counts,
            }
        )

    return warnings


def parse_folder(
    folder_path: str | Path,
    encoding: str | None = None,
    recursive: bool = False,
    stop_on_error: bool = False,
    log: bool = True,
) -> FolderParseResult:
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder does not exist or is not a directory: {folder}")

    files = iter_txt_files(folder, recursive=recursive)
    dataframes = []
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    files_ok = 0
    started_at = time.monotonic()

    if log:
        mode = "recursive" if recursive else "non-recursive"
        print(f"[start] folder={folder} txt_files={len(files)} mode={mode}", flush=True)

    for file_number, path in enumerate(files, start=1):
        if log:
            print(f"[{file_number}/{len(files)}] parsing {path}", flush=True)

        try:
            result = parse_document(read_table_text(path, encoding=encoding))
            file_dataframes = parsed_result_to_dataframes(
                result,
                source_path=path,
                file_number=file_number,
            )
            dataframes.extend(file_dataframes)
            files_ok += 1

            file_warnings = []
            if not result.tables:
                file_warnings.append(
                    {
                        "file_number": file_number,
                        "source_file": path.name,
                        "source_path": str(path),
                        "table_number": None,
                        "kind": "no_tables",
                        "details": {
                            "physical_rows": result.debug.physical_rows,
                            "skipped_leading_rows": result.debug.skipped_leading_rows,
                            "headers_found": result.debug.headers_found,
                        },
                    }
                )

            for table_number, table in enumerate(result.tables, start=1):
                file_warnings.extend(
                    table_warning_records(
                        table,
                        file_number=file_number,
                        source_path=path,
                        table_number=table_number,
                    )
                )
            warnings.extend(file_warnings)

            if log:
                rows_total = sum(len(table.rows) for table in result.tables)
                warning_text = f" warnings={len(file_warnings)}" if file_warnings else ""
                print(
                    f"[{file_number}/{len(files)}] ok tables={len(result.tables)} "
                    f"rows={rows_total}{warning_text}",
                    flush=True,
                )
                for warning in file_warnings:
                    table_label = (
                        warning["table_number"]
                        if warning["table_number"] is not None
                        else "-"
                    )
                    print(
                        f"[warning] file={path.name} table={table_label} "
                        f"{warning['kind']}: {warning['details']}",
                        flush=True,
                    )

        except Exception as exc:
            error = {
                "file_number": file_number,
                "source_file": path.name,
                "source_path": str(path),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            errors.append(error)

            if log:
                print(
                    f"[{file_number}/{len(files)}] ERROR {type(exc).__name__}: {exc}",
                    flush=True,
                )
                print(error["traceback"], flush=True)

            if stop_on_error:
                raise

    elapsed = time.monotonic() - started_at
    result = FolderParseResult(
        dataframes=dataframes,
        errors=errors,
        warnings=warnings,
        files_total=len(files),
        files_ok=files_ok,
        files_failed=len(errors),
        tables_total=len(dataframes),
    )

    if log:
        print(
            f"[done] files_ok={result.files_ok}/{result.files_total} "
            f"files_failed={result.files_failed} tables={result.tables_total} "
            f"warnings={len(result.warnings)} elapsed_sec={elapsed:.1f}",
            flush=True,
        )

    return result


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
    parser.add_argument("--encoding")
    args = parser.parse_args()

    result = parse_document(read_table_text(args.path, encoding=args.encoding))
    print(json.dumps(summarize_result(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
