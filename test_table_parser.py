import csv
import io
import json
import unittest
from contextlib import redirect_stdout
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from table_parser import (
    parse_document,
    parse_folder,
    parse_folder_fast,
    parse_tables_to_dataframes,
    read_table_text,
)


class TableParserTests(unittest.TestCase):
    def test_splits_tables_and_joins_wrapped_rows_without_modification(self) -> None:
        result = parse_document(
            "\n".join(
                [
                    "before",
                    "\t Вид\tA\tB",
                    "",
                    "PM1\tone\ttwo",
                    " wrapped",
                    "\t Вид\tC",
                    "PM2\tthree",
                ]
            )
        )

        self.assertEqual(result.debug.skipped_leading_rows, 1)
        self.assertEqual(result.debug.headers_found, 2)
        self.assertEqual(len(result.tables), 2)

        first = result.tables[0]
        self.assertEqual(first.header, ["Вид", "A", "B"])
        self.assertEqual(first.debug.blank_rows_removed, 1)
        self.assertEqual(first.debug.continuation_rows_joined, 1)
        self.assertEqual(first.rows[0].line_numbers, [4, 5])
        self.assertEqual(first.rows[0].raw, "PM1\tone\ttwo wrapped")

    def test_parses_rows_with_header_tab_layout_and_right_padding(self) -> None:
        result = parse_document(
            "\n".join(
                [
                    "\t Вид\tA\tB\tC\tD",
                    "PM1\tone\t\ttwo\tthree",
                    "PM2\tone\ttwo\t\tthree",
                    "PM3\tone\ttwo",
                ]
            )
        )

        table = result.tables[0]
        self.assertEqual(table.separator_widths, [1, 1, 1, 1])
        self.assertEqual(table.rows[0].cells, ["PM1", "one", "", "two", "three"])
        self.assertEqual(table.rows[1].cells, ["PM2", "one", "two", "three", ""])
        self.assertEqual(table.rows[2].cells, ["PM3", "one", "two", "", ""])
        self.assertEqual(table.debug.rows_using_special_separator_width, 1)
        self.assertEqual(table.debug.rows_with_short_right_side, 2)

    def test_parses_anonymized_reference_shape(self) -> None:
        result = parse_document(read_table_text(Path("table_format.txt")))

        self.assertEqual(len(result.tables), 3)
        self.assertEqual([table.header_line_number for table in result.tables], [2, 66, 130])
        self.assertEqual([len(table.rows) for table in result.tables], [60, 60, 15])
        self.assertEqual([len(table.header) for table in result.tables], [21, 21, 21])
        self.assertFalse(any(row.issues for table in result.tables for row in table.rows))

    def test_read_table_text_handles_utf8_bom_and_cp1251_files(self) -> None:
        with TemporaryDirectory() as tmpdir:
            utf8_path = Path(tmpdir) / "utf8.txt"
            cp1251_path = Path(tmpdir) / "cp1251.txt"
            utf8_path.write_bytes("\ufeff\t Вид\tA\nPM1\tone\n".encode("utf-8"))
            cp1251_path.write_bytes("\t Вид\tИмя\nPM1\tОдин\n".encode("cp1251"))

            self.assertTrue(read_table_text(utf8_path).startswith("\t Вид"))
            self.assertIn("Имя", read_table_text(cp1251_path))

    def test_drops_structural_leading_tab_when_parsing_data_cells(self) -> None:
        result = parse_document(
            "\n".join(
                [
                    "\t H1\tH2\tH3",
                    "\tAA1\tone\ttwo",
                ]
            )
        )

        self.assertEqual(result.tables[0].header, ["H1", "H2", "H3"])
        self.assertEqual(result.tables[0].rows[0].cells, ["AA1", "one", "two"])

    def test_anonymized_prefix_does_not_start_real_marker_table_rows(self) -> None:
        result = parse_document(
            "\n".join(
                [
                    "\t Вид\tA",
                    "PM1\tone",
                    "AA continuation",
                ]
            )
        )

        self.assertEqual(len(result.tables[0].rows), 1)
        self.assertEqual(result.tables[0].rows[0].line_numbers, [2, 3])

    def test_real_pm_marker_can_have_structural_prefix_chars(self) -> None:
        result = parse_document(
            "\n".join(
                [
                    "\t Вид\tA",
                    "\t PM1\tone",
                    "\t\xa0PM2\ttwo",
                ]
            )
        )

        self.assertEqual(len(result.tables[0].rows), 2)
        self.assertEqual(result.tables[0].rows[0].line_numbers, [2])
        self.assertEqual(result.tables[0].rows[1].line_numbers, [3])

    @unittest.skipIf(find_spec("pandas") is None, "pandas is not installed")
    def test_returns_one_dataframe_per_table(self) -> None:
        text = "\n".join(
            [
                "before",
                "\t Вид\tA\tB",
                "PM1\tone\ttwo",
                "wrapped",
                "\t Вид\tC",
                "PM2\tthree",
            ]
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "input.txt"
            path.write_text(text)

            tables = parse_tables_to_dataframes(path)

        self.assertEqual(len(tables), 2)
        self.assertEqual(
            list(tables[0].columns),
            ["table_number", "row_number", "physical_row_numbers", "Вид", "A", "B"],
        )
        self.assertEqual(tables[0].loc[0, "table_number"], 1)
        self.assertEqual(tables[0].loc[0, "row_number"], 1)
        self.assertEqual(tables[0].loc[0, "physical_row_numbers"], [3, 4])
        self.assertEqual(tables[0].loc[0, "B"], "twowrapped")
        self.assertEqual(tables[1].loc[0, "table_number"], 2)

    @unittest.skipIf(find_spec("pandas") is None, "pandas is not installed")
    def test_parse_folder_logs_progress_and_captures_file_errors(self) -> None:
        with TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            bad_path = folder / "bad.txt"
            good_path = folder / "good.txt"
            bad_path.write_text("bad")
            good_path.write_text("\t Вид\tA\nPM1\tone\n")

            original_read_table_text = read_table_text

            def read_with_failure(path: str | Path, encoding: str | None = None) -> str:
                if Path(path).name == "bad.txt":
                    raise ValueError("forced parse failure")
                return original_read_table_text(path, encoding=encoding)

            output = io.StringIO()
            with patch("table_parser.read_table_text", side_effect=read_with_failure):
                with redirect_stdout(output):
                    result = parse_folder(folder)

        self.assertEqual(result.files_total, 2)
        self.assertEqual(result.files_ok, 1)
        self.assertEqual(result.files_failed, 1)
        self.assertEqual(result.tables_total, 1)
        self.assertEqual(result.errors[0]["source_file"], "bad.txt")
        self.assertEqual(result.errors[0]["error_type"], "ValueError")
        self.assertIn("forced parse failure", result.errors[0]["traceback"])
        self.assertIn("[start]", output.getvalue())
        self.assertIn("[1/2] parsing", output.getvalue())
        self.assertIn("ERROR ValueError", output.getvalue())
        self.assertIn("[done]", output.getvalue())
        self.assertEqual(
            list(result.dataframes[0].columns[:6]),
            [
                "file_number",
                "source_file",
                "source_path",
                "table_number",
                "row_number",
                "physical_row_numbers",
            ],
        )
        self.assertEqual(result.dataframes[0].loc[0, "source_file"], "good.txt")

    @unittest.skipIf(find_spec("pandas") is None, "pandas is not installed")
    def test_parse_folder_can_throttle_progress_logs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            for index in range(4):
                (folder / f"{index}.txt").write_text("\t Вид\tA\nPM1\tone\n")

            output = io.StringIO()
            with redirect_stdout(output):
                result = parse_folder(folder, progress_every=3)

        self.assertEqual(result.files_ok, 4)
        log_text = output.getvalue()
        self.assertIn("[1/4] parsing", log_text)
        self.assertNotIn("[2/4] parsing", log_text)
        self.assertIn("[3/4] parsing", log_text)
        self.assertIn("[4/4] parsing", log_text)

    def test_parse_folder_fast_streams_one_output_per_file(self) -> None:
        with TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir) / "input"
            output = Path(tmpdir) / "output"
            folder.mkdir()
            (folder / "a.txt").write_text(
                "\n".join(
                    [
                        "ignore",
                        "\t Вид\tA",
                        "PM1\tone",
                        "wrapped",
                        "",
                        "",
                        "\t Вид\tB",
                        "PM2\ttwo",
                    ]
                ),
                encoding="utf-8",
            )
            (folder / "b.txt").write_text("\t Вид\tA\nPM3\tthree\n", encoding="utf-8")

            result = parse_folder_fast(folder, output_dir=output, workers=1, log=False)

            self.assertEqual(result.files_total, 2)
            self.assertEqual(result.files_ok, 2)
            self.assertEqual(result.files_failed, 0)
            self.assertEqual(result.rows_written, 3)
            self.assertEqual(len(result.results), 2)
            self.assertTrue(Path(result.summary_path).exists())

            first_result = result.results[0]
            with Path(first_result.output_rows_path).open(encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f, delimiter="\t"))
            self.assertEqual(rows[0][:5], ["1", "1", "3,4", "1", "PM1"])
            self.assertEqual(rows[0][5], "onewrapped")
            self.assertEqual(rows[1][:6], ["1", "2", "8", "2", "PM2", "two"])

            headers_payload = json.loads(
                Path(first_result.output_headers_path).read_text(encoding="utf-8")
            )
            self.assertEqual(headers_payload["max_cells"], 2)
            self.assertEqual(len(headers_payload["headers"]), 2)
            self.assertEqual(headers_payload["headers"][0]["columns"], ["Вид", "A"])
            self.assertEqual(headers_payload["headers"][1]["columns"], ["Вид", "B"])


if __name__ == "__main__":
    unittest.main()
