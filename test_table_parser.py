import unittest
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory

from table_parser import parse_document, parse_tables_to_dataframes


class TableParserTests(unittest.TestCase):
    def test_splits_tables_and_joins_wrapped_rows_without_modification(self) -> None:
        result = parse_document(
            "\n".join(
                [
                    "before",
                    "\t Вид\tA\tB",
                    "",
                    "РМ1\tone\ttwo",
                    " wrapped",
                    "\t Вид\tC",
                    "РМ2\tthree",
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
        self.assertEqual(first.rows[0].raw, "РМ1\tone\ttwo wrapped")

    def test_parses_rows_with_header_tab_layout_and_right_padding(self) -> None:
        result = parse_document(
            "\n".join(
                [
                    "\t Вид\tA\tB\tC\tD",
                    "РМ1\tone\t\ttwo\tthree",
                    "РМ2\tone\ttwo\t\tthree",
                    "РМ3\tone\ttwo",
                ]
            )
        )

        table = result.tables[0]
        self.assertEqual(table.separator_widths, [1, 1, 1, 1])
        self.assertEqual(table.rows[0].cells, ["РМ1", "one", "", "two", "three"])
        self.assertEqual(table.rows[1].cells, ["РМ2", "one", "two", "three", ""])
        self.assertEqual(table.rows[2].cells, ["РМ3", "one", "two", "", ""])
        self.assertEqual(table.debug.rows_using_special_separator_width, 1)
        self.assertEqual(table.debug.rows_with_short_right_side, 2)

    @unittest.skipIf(find_spec("pandas") is None, "pandas is not installed")
    def test_returns_one_dataframe_per_table(self) -> None:
        text = "\n".join(
            [
                "before",
                "\t Вид\tA\tB",
                "РМ1\tone\ttwo",
                "wrapped",
                "\t Вид\tC",
                "РМ2\tthree",
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


if __name__ == "__main__":
    unittest.main()
