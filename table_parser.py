from __future__ import annotations

import argparse
import base64
import binascii
import csv
import json
import os
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
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


@dataclass
class FastFileParseResult:
    file_number: int
    source_file: str
    source_path: str
    output_rows_path: str
    output_headers_path: str
    encoding: str
    bytes_read: int
    physical_rows: int
    rows_written: int
    headers_found: int
    header_variants: int
    skipped_leading_rows: int
    blank_rows_removed: int
    continuation_rows_joined: int
    orphan_continuation_rows: int
    rows_with_short_right_side: int
    rows_with_extra_columns: int
    rows_with_delimiter_issues: int
    rows_with_issues: int
    elapsed_sec: float
    issue_samples: list[dict[str, Any]]


@dataclass
class FastFolderParseResult:
    output_dir: str
    files_total: int
    files_ok: int
    files_failed: int
    rows_written: int
    elapsed_sec: float
    results: list[FastFileParseResult]
    errors: list[dict[str, Any]]
    summary_path: str


@dataclass
class SeparatorLayout:
    separator_widths: tuple[int, ...]
    resolved_counts: tuple[dict[int, int], ...]
    min_widths: tuple[int, ...]


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


def is_header_raw(raw: str) -> bool:
    return raw.lstrip("\ufeff\u200b\u200c\u200d").startswith(HEADER_PREFIX)


def is_header_row(row: PhysicalRow) -> bool:
    return is_header_raw(row.raw)


def row_content_start(raw: str, drop_leading_tab: bool = False) -> int:
    start = 1 if drop_leading_tab and raw.startswith("\t") else 0
    raw_length = len(raw)

    while start < raw_length and raw[start] in ROW_START_IGNORED_CHARS:
        start += 1

    return start


def first_cell_for_classification(raw: str, drop_leading_tab: bool = False) -> str:
    start = row_content_start(raw, drop_leading_tab=drop_leading_tab)
    end = raw.find("\t", start)
    if end == -1:
        end = len(raw)

    return raw[start:end].strip(ROW_START_IGNORED_CHARS)


def is_data_row(
    raw: str,
    allow_anonymized_prefix: bool = False,
    drop_leading_tab: bool = False,
) -> bool:
    if raw[: len(DATA_ROW_PREFIX)].upper() == DATA_ROW_PREFIX:
        return True

    start = row_content_start(raw, drop_leading_tab=drop_leading_tab)

    if raw[start : start + len(DATA_ROW_PREFIX)].upper() == DATA_ROW_PREFIX:
        return True

    if not allow_anonymized_prefix:
        return False

    if not raw.startswith(ANONYMIZED_DATA_ROW_PREFIX, start):
        return False

    end = raw.find("\t", start)
    if end == -1:
        end = len(raw)

    return "1" in raw[start:end]


def is_fallback_header_raw(raw: str) -> bool:
    return raw.count("\t") >= MIN_HEADER_TABS and not is_data_row(
        raw,
        allow_anonymized_prefix=True,
    )


def is_fallback_header_row(row: PhysicalRow) -> bool:
    return is_fallback_header_raw(row.raw)


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


def build_separator_layout(separator_widths: list[int]) -> SeparatorLayout:
    return build_separator_layout_cached(tuple(separator_widths))


@lru_cache(maxsize=128)
def build_separator_layout_cached(widths: tuple[int, ...]) -> SeparatorLayout:
    resolved_counts: list[dict[int, int]] = []
    min_widths = tuple(
        min(separator_width_options(index, expected_width))
        for index, expected_width in enumerate(widths)
    )

    for start_index in range(len(widths)):
        counts_by_run_width: dict[int, int] = {}
        sums = {0}
        for end_index in range(start_index, len(widths)):
            sums = {
                total + option
                for total in sums
                for option in separator_width_options(end_index, widths[end_index])
            }
            separator_count = end_index - start_index + 1
            for run_width in sums:
                counts_by_run_width.setdefault(run_width, separator_count)
        resolved_counts.append(counts_by_run_width)

    return SeparatorLayout(
        separator_widths=widths,
        resolved_counts=tuple(resolved_counts),
        min_widths=min_widths,
    )


def parse_cells(
    raw: str,
    header_width: int,
    layout: SeparatorLayout,
    drop_leading_tab: bool = False,
) -> tuple[list[str], list[str], bool, bool]:
    start = 1 if drop_leading_tab and raw.startswith("\t") else 0
    raw_length = len(raw)
    cells: list[str] = []
    separator_index = 0
    issues: list[str] = []
    used_special_separator_width = False
    separator_widths = layout.separator_widths
    resolved_counts = layout.resolved_counts
    min_widths = layout.min_widths
    separator_widths_count = len(separator_widths)
    find_tab = raw.find

    position = start
    while True:
        tab_start = find_tab("\t", position)
        if tab_start == -1:
            cells.append(raw[position:])
            break

        cells.append(raw[position:tab_start])
        tab_end = tab_start + 1
        while tab_end < raw_length and raw[tab_end] == "\t":
            tab_end += 1
        run_width = tab_end - tab_start

        if separator_index >= separator_widths_count:
            issues.append(f"extra delimiter run of {run_width} tabs after header columns")
            position = tab_end
            continue

        separator_count = resolved_counts[separator_index].get(run_width)
        if separator_count is None:
            total_width = 0
            separator_count = 0
            for index in range(separator_index, separator_widths_count):
                total_width += min_widths[index]
                separator_count += 1
                if total_width >= run_width:
                    break
            if separator_count == 0:
                separator_count = 1
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

        if separator_count > 1:
            cells.extend([""] * (separator_count - 1))
        separator_index += separator_count
        position = tab_end

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
        separator_layout = build_separator_layout(separator_widths)
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
                separator_layout,
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
        header_len = len(table.header)
        base_values = meta_values + [table_number]
        records_append = records.append

        for row_number, row in enumerate(table.rows, start=1):
            cells = row.cells if len(row.cells) == header_len else row.cells[:header_len]
            records_append(base_values + [row_number, row.line_numbers] + cells)

        dataframes.append(pd.DataFrame(records, columns=columns, dtype=object))

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
    progress_every: int = 1,
) -> FolderParseResult:
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder does not exist or is not a directory: {folder}")
    if progress_every < 1:
        raise ValueError("progress_every must be 1 or greater")

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
        show_progress = log and (
            progress_every == 1
            or file_number == 1
            or file_number == len(files)
            or file_number % progress_every == 0
        )

        if show_progress:
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

            if show_progress or (log and file_warnings):
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


def detect_text_encoding(
    path: str | Path,
    encoding: str | None = None,
    sample_bytes: int = 1024 * 1024,
) -> str:
    if encoding is not None:
        return encoding

    path = Path(path)
    with path.open("rb") as sample_file:
        sample = sample_file.read(sample_bytes)
    if not sample:
        return DEFAULT_TEXT_ENCODINGS[0]

    last_error: UnicodeDecodeError | None = None
    for candidate_encoding in DEFAULT_TEXT_ENCODINGS:
        try:
            sample.decode(candidate_encoding)
            return candidate_encoding
        except UnicodeDecodeError as exc:
            last_error = exc

    assert last_error is not None
    last_error.add_note(f"Tried encodings: {', '.join(DEFAULT_TEXT_ENCODINGS)}")
    raise last_error


def safe_output_stem(path: Path, file_number: int) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._")
    if not safe_stem:
        safe_stem = "table"
    return f"{file_number:04d}_{safe_stem[:80]}"


def fast_result_to_dict(result: FastFileParseResult) -> dict[str, Any]:
    return {
        "file_number": result.file_number,
        "source_file": result.source_file,
        "source_path": result.source_path,
        "output_rows_path": result.output_rows_path,
        "output_headers_path": result.output_headers_path,
        "encoding": result.encoding,
        "bytes_read": result.bytes_read,
        "physical_rows": result.physical_rows,
        "rows_written": result.rows_written,
        "headers_found": result.headers_found,
        "header_variants": result.header_variants,
        "skipped_leading_rows": result.skipped_leading_rows,
        "blank_rows_removed": result.blank_rows_removed,
        "continuation_rows_joined": result.continuation_rows_joined,
        "orphan_continuation_rows": result.orphan_continuation_rows,
        "rows_with_short_right_side": result.rows_with_short_right_side,
        "rows_with_extra_columns": result.rows_with_extra_columns,
        "rows_with_delimiter_issues": result.rows_with_delimiter_issues,
        "rows_with_issues": result.rows_with_issues,
        "elapsed_sec": result.elapsed_sec,
        "issue_samples": result.issue_samples,
    }


def parse_file_fast(
    path: str | Path,
    output_dir: str | Path,
    file_number: int = 1,
    files_total: int = 1,
    encoding: str | None = None,
    sample_bytes: int = 1024 * 1024,
    progress_mb: int = 256,
    log: bool = True,
) -> FastFileParseResult:
    """Stream one large txt file to one TSV row file plus a header map JSON.

    This path is for production-scale files. It does not build pandas objects
    and it does not read the whole input into memory.
    """
    path = Path(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_encoding = detect_text_encoding(
        path,
        encoding=encoding,
        sample_bytes=sample_bytes,
    )
    output_stem = safe_output_stem(path, file_number)
    output_rows_path = output_dir / f"{output_stem}.rows.tsv"
    output_headers_path = output_dir / f"{output_stem}.headers.json"

    started_at = time.monotonic()
    file_size = path.stat().st_size
    progress_step = max(progress_mb, 1) * 1024 * 1024
    next_progress_at = progress_step

    bytes_read = 0
    physical_rows = 0
    rows_written = 0
    headers_found = 0
    skipped_leading_rows = 0
    blank_rows_removed = 0
    continuation_rows_joined = 0
    orphan_continuation_rows = 0
    rows_with_short_right_side = 0
    rows_with_extra_columns = 0
    rows_with_delimiter_issues = 0
    rows_with_issues = 0
    max_cells = 0
    blank_run = 0
    issue_samples: list[dict[str, Any]] = []

    header_by_key: dict[tuple[tuple[str, ...], tuple[int, ...]], int] = {}
    header_records: list[dict[str, Any]] = []
    current_header: list[str] | None = None
    current_layout: SeparatorLayout | None = None
    current_header_id = 0
    drop_leading_tab = False
    allow_anonymized_prefix = False

    pending_line_numbers: list[int] | None = None
    pending_parts: list[str] | None = None
    pending_issues: list[str] = []

    if log:
        print(
            f"[file {file_number}/{files_total}] start {path} "
            f"size_mb={file_size / 1024 / 1024:.1f} encoding={selected_encoding}",
            flush=True,
        )

    def set_header(raw: str, line_number: int) -> None:
        nonlocal current_header
        nonlocal current_layout
        nonlocal current_header_id
        nonlocal drop_leading_tab
        nonlocal allow_anonymized_prefix
        nonlocal headers_found

        header, separator_widths = parse_header(raw)
        key = (tuple(header), tuple(separator_widths))
        header_id = header_by_key.get(key)
        if header_id is None:
            header_id = len(header_records) + 1
            header_by_key[key] = header_id
            header_records.append(
                {
                    "header_number": header_id,
                    "first_line_number": line_number,
                    "columns": header,
                    "separator_widths": separator_widths,
                    "occurrences": 1,
                }
            )
        else:
            header_records[header_id - 1]["occurrences"] += 1

        current_header = header
        current_layout = build_separator_layout(separator_widths)
        current_header_id = header_id
        drop_leading_tab = raw.startswith("\t")
        allow_anonymized_prefix = not is_header_raw(raw)
        headers_found += 1

    with path.open("rb") as input_file, output_rows_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")

        def finalize_pending_row() -> None:
            nonlocal pending_line_numbers
            nonlocal pending_parts
            nonlocal pending_issues
            nonlocal rows_written
            nonlocal rows_with_short_right_side
            nonlocal rows_with_extra_columns
            nonlocal rows_with_delimiter_issues
            nonlocal rows_with_issues
            nonlocal max_cells

            if pending_line_numbers is None or pending_parts is None:
                return

            if current_header is None or current_layout is None:
                pending_line_numbers = None
                pending_parts = None
                pending_issues = []
                return

            raw = "".join(pending_parts)
            cells, cell_issues, short_right_side, _ = parse_cells(
                raw,
                len(current_header),
                current_layout,
                drop_leading_tab=drop_leading_tab,
            )
            issues = pending_issues + cell_issues

            rows_written += 1
            max_cells = max(max_cells, len(cells))
            if short_right_side:
                rows_with_short_right_side += 1
            if len(cells) > len(current_header):
                rows_with_extra_columns += 1
            if cell_issues:
                rows_with_delimiter_issues += 1
            if issues:
                rows_with_issues += 1
                if len(issue_samples) < 20:
                    issue_samples.append(
                        {
                            "row_number": rows_written,
                            "line_numbers": pending_line_numbers.copy(),
                            "issues": issues,
                        }
                    )

            writer.writerow(
                [
                    1,
                    rows_written,
                    ",".join(str(number) for number in pending_line_numbers),
                    current_header_id,
                    *cells,
                ]
            )

            pending_line_numbers = None
            pending_parts = None
            pending_issues = []

        for line_bytes in input_file:
            bytes_read += len(line_bytes)
            physical_rows += 1
            raw = line_bytes.decode(selected_encoding).rstrip("\r\n")

            if log and bytes_read >= next_progress_at:
                elapsed = max(time.monotonic() - started_at, 0.001)
                percent = (bytes_read / file_size * 100) if file_size else 100.0
                print(
                    f"[file {file_number}/{files_total}] {path.name} "
                    f"{bytes_read / 1024 / 1024:.0f}/{file_size / 1024 / 1024:.0f} MB "
                    f"({percent:.1f}%) rows={rows_written} "
                    f"headers={headers_found} speed_mb_s={bytes_read / 1024 / 1024 / elapsed:.1f}",
                    flush=True,
                )
                while next_progress_at <= bytes_read:
                    next_progress_at += progress_step

            if not raw or raw.isspace():
                blank_run += 1
                if current_header is None:
                    skipped_leading_rows += 1
                else:
                    blank_rows_removed += 1
                continue

            starts_header_by_marker = is_header_raw(raw)
            starts_first_header_by_shape = (
                current_header is None and is_fallback_header_raw(raw)
            )
            starts_next_header_by_shape = (
                current_header is not None
                and blank_run >= TABLE_GAP
                and is_fallback_header_raw(raw)
            )

            if (
                starts_header_by_marker
                or starts_first_header_by_shape
                or starts_next_header_by_shape
            ):
                finalize_pending_row()
                set_header(raw, physical_rows)
                blank_run = 0
                continue

            if current_header is None:
                skipped_leading_rows += 1
                blank_run = 0
                continue

            if is_data_row(
                raw,
                allow_anonymized_prefix=allow_anonymized_prefix,
                drop_leading_tab=drop_leading_tab,
            ):
                finalize_pending_row()
                pending_line_numbers = [physical_rows]
                pending_parts = [raw]
                pending_issues = []
            elif pending_line_numbers is None or pending_parts is None:
                orphan_continuation_rows += 1
                pending_line_numbers = [physical_rows]
                pending_parts = [raw]
                pending_issues = ["non-PM row has no previous data row to join"]
            else:
                continuation_rows_joined += 1
                pending_line_numbers.append(physical_rows)
                pending_parts.append(raw)

            blank_run = 0

        finalize_pending_row()

    elapsed_sec = time.monotonic() - started_at
    result = FastFileParseResult(
        file_number=file_number,
        source_file=path.name,
        source_path=str(path),
        output_rows_path=str(output_rows_path),
        output_headers_path=str(output_headers_path),
        encoding=selected_encoding,
        bytes_read=bytes_read,
        physical_rows=physical_rows,
        rows_written=rows_written,
        headers_found=headers_found,
        header_variants=len(header_records),
        skipped_leading_rows=skipped_leading_rows,
        blank_rows_removed=blank_rows_removed,
        continuation_rows_joined=continuation_rows_joined,
        orphan_continuation_rows=orphan_continuation_rows,
        rows_with_short_right_side=rows_with_short_right_side,
        rows_with_extra_columns=rows_with_extra_columns,
        rows_with_delimiter_issues=rows_with_delimiter_issues,
        rows_with_issues=rows_with_issues,
        elapsed_sec=elapsed_sec,
        issue_samples=issue_samples,
    )

    headers_payload = {
        "source_file": path.name,
        "source_path": str(path),
        "rows_path": str(output_rows_path),
        "rows_tsv_layout": [
            "table_number",
            "row_number",
            "physical_row_numbers",
            "header_number",
            "cell_1",
            "...",
        ],
        "max_cells": max_cells,
        "headers": header_records,
        "summary": fast_result_to_dict(result),
    }
    output_headers_path.write_text(
        json.dumps(headers_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if log:
        print(
            f"[file {file_number}/{files_total}] done {path.name} "
            f"rows={rows_written} headers={headers_found} "
            f"header_variants={len(header_records)} "
            f"issue_rows={rows_with_issues} elapsed_sec={elapsed_sec:.1f}",
            flush=True,
        )

    return result


def parse_folder_fast(
    folder_path: str | Path,
    output_dir: str | Path | None = None,
    encoding: str | None = None,
    recursive: bool = False,
    workers: int | None = None,
    progress_mb: int = 256,
    sample_bytes: int = 1024 * 1024,
    log: bool = True,
    stop_on_error: bool = False,
) -> FastFolderParseResult:
    """Parse all txt files in a folder with one output table per input file."""
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder does not exist or is not a directory: {folder}")

    files = iter_txt_files(folder, recursive=recursive)
    if output_dir is None:
        output_dir = folder / "parsed_tables_fast"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if workers is None:
        workers = min(len(files), os.cpu_count() or 1, 4) if files else 1
    workers = max(workers, 1)

    started_at = time.monotonic()
    if log:
        mode = "recursive" if recursive else "non-recursive"
        print(
            f"[fast-start] folder={folder} txt_files={len(files)} "
            f"workers={workers} mode={mode} output_dir={output_path}",
            flush=True,
        )

    results: list[FastFileParseResult] = []
    errors: list[dict[str, Any]] = []

    jobs = [
        {
            "path": str(path),
            "output_dir": str(output_path),
            "file_number": file_number,
            "files_total": len(files),
            "encoding": encoding,
            "sample_bytes": sample_bytes,
            "progress_mb": progress_mb,
            "log": log,
        }
        for file_number, path in enumerate(files, start=1)
    ]

    if workers == 1:
        for job in jobs:
            try:
                results.append(parse_file_fast(**job))
            except Exception as exc:
                error = {
                    "file_number": job["file_number"],
                    "source_path": job["path"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                errors.append(error)
                if log:
                    print(
                        f"[fast-error] file={job['path']} "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    print(error["traceback"], flush=True)
                if stop_on_error:
                    raise
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_job = {
                executor.submit(parse_file_fast, **job): job
                for job in jobs
            }
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    error = {
                        "file_number": job["file_number"],
                        "source_path": job["path"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    errors.append(error)
                    if log:
                        print(
                            f"[fast-error] file={job['path']} "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        print(error["traceback"], flush=True)
                    if stop_on_error:
                        raise

    results.sort(key=lambda result: result.file_number)
    errors.sort(key=lambda error: error["file_number"])
    elapsed_sec = time.monotonic() - started_at
    summary_path = output_path / "parse_summary.json"
    folder_result = FastFolderParseResult(
        output_dir=str(output_path),
        files_total=len(files),
        files_ok=len(results),
        files_failed=len(errors),
        rows_written=sum(result.rows_written for result in results),
        elapsed_sec=elapsed_sec,
        results=results,
        errors=errors,
        summary_path=str(summary_path),
    )

    summary_payload = {
        "output_dir": str(output_path),
        "files_total": folder_result.files_total,
        "files_ok": folder_result.files_ok,
        "files_failed": folder_result.files_failed,
        "rows_written": folder_result.rows_written,
        "elapsed_sec": folder_result.elapsed_sec,
        "results": [fast_result_to_dict(result) for result in results],
        "errors": errors,
    }
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if log:
        print(
            f"[fast-done] files_ok={folder_result.files_ok}/{folder_result.files_total} "
            f"files_failed={folder_result.files_failed} rows={folder_result.rows_written} "
            f"elapsed_sec={elapsed_sec:.1f} summary={summary_path}",
            flush=True,
        )

    return folder_result


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
    parser.add_argument(
        "--fast-folder",
        action="store_true",
        help="stream all txt files in a folder and write one output table per file",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--progress-mb", type=int, default=256)
    parser.add_argument("--sample-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    if args.fast_folder:
        result = parse_folder_fast(
            args.path,
            output_dir=args.output_dir,
            encoding=args.encoding,
            recursive=args.recursive,
            workers=args.workers,
            progress_mb=args.progress_mb,
            sample_bytes=args.sample_bytes,
            stop_on_error=args.stop_on_error,
        )
        print(
            json.dumps(
                {
                    "output_dir": result.output_dir,
                    "files_total": result.files_total,
                    "files_ok": result.files_ok,
                    "files_failed": result.files_failed,
                    "rows_written": result.rows_written,
                    "elapsed_sec": result.elapsed_sec,
                    "summary_path": result.summary_path,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    result = parse_document(read_table_text(args.path, encoding=args.encoding))
    print(json.dumps(summarize_result(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
