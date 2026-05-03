# Table parser

Parser for large txt table files with tab-delimited columns and wrapped data rows.

Production mode is `parse_folder_fast`: it streams files line by line, avoids pandas, avoids reading whole 1 GB files into memory, and can process several files in parallel.

## Install

Fast mode uses only the Python standard library. Install requirements only if you also need the older pandas DataFrame helpers or tests:

```powershell
pip install -r requirements.txt
```

## Fast production run

Run this from PowerShell:

```powershell
python table_parser.py "C:\path\to\txt_folder" --fast-folder --output-dir "C:\path\to\parsed" --workers 4 --encoding cp1251 --progress-mb 256
```

For UTF-8 input, use:

```powershell
python table_parser.py "C:\path\to\txt_folder" --fast-folder --output-dir "C:\path\to\parsed" --workers 4 --encoding utf-8-sig --progress-mb 256
```

If encoding is omitted, the parser samples each file and tries `utf-8-sig`, `cp1251`, then `cp1252`.

Recommended worker count:

- Slow HDD or external disk: `--workers 2`
- Normal SSD: `--workers 4`
- Fast NVMe: try `--workers 6` or `--workers 9`

Progress is printed to the console. `--progress-mb 256` means each worker prints status after every 256 MB read.

## Jupyter usage

```python
from table_parser import parse_folder_fast

result = parse_folder_fast(
    r"C:\path\to\txt_folder",
    output_dir=r"C:\path\to\parsed",
    workers=4,
    encoding="cp1251",
    progress_mb=256,
)

result.summary_path
```

## Fast output files

For each input txt file, fast mode writes:

- `NNNN_filename.rows.tsv`
- `NNNN_filename.headers.json`

The folder also gets:

- `parse_summary.json`

Rows TSV layout:

```text
table_number    row_number    physical_row_numbers    header_number    cell_1    cell_2    ...
```

Notes:

- Fast mode creates one output table per input file.
- `physical_row_numbers` preserves original file line numbers. Wrapped rows are stored as comma-separated line numbers.
- `header_number` points to the matching header variant in the file's `headers.json`.
- If headers inside one file differ slightly, all variants are stored in `headers.json`.
- Parsing warnings and sample row issues are stored in `parse_summary.json` and each file's `headers.json`.

## Small/debug pandas API

For small files and debugging:

```python
from table_parser import parse_file, parse_folder

dfs = parse_file(r"C:\path\to\one_file.txt", encoding="cp1251")
folder_result = parse_folder(r"C:\path\to\txt_folder", encoding="cp1251")
```

These functions return pandas DataFrames and are not recommended for 1 GB production files.

## Anonymizer

```python
from anonymize_text import anonymize_file

encoded = anonymize_file(
    r"C:\path\to\source.txt",
    output_path=r"C:\path\to\source.anonymized.txt",
    encoding="cp1251",
)
```

The anonymizer replaces letters with `A`, digits with `1`, preserves whitespace and punctuation, prints the normalized pre-base64 text, and returns base64 text.

## Tests

```powershell
python -m unittest -v
```
