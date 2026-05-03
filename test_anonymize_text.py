import base64
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from anonymize_text import anonymize_file, normalize_and_encode, normalize_text


class AnonymizeTextTests(unittest.TestCase):
    def test_normalizes_unicode_letters_digits_and_preserves_other_chars(self) -> None:
        text = "Hello Привет 123!\t@#\n"

        self.assertEqual(normalize_text(text), "AAAAA AAAAAA 111!\t@#\n")

    def test_normalize_and_encode_returns_base64_of_normalized_text(self) -> None:
        text = "Aя9 -_\t\n"
        encoded = normalize_and_encode(text)

        self.assertEqual(base64.b64decode(encoded).decode("utf-8"), "AA1 -_\t\n")

    def test_anonymize_file_returns_and_optionally_writes_encoded_text(self) -> None:
        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.txt"
            output_path = Path(tmpdir) / "output.b64"
            input_path.write_text("Test 42")

            encoded = anonymize_file(input_path, output_path)

            self.assertEqual(output_path.read_text(), encoded)
            self.assertEqual(base64.b64decode(encoded).decode("utf-8"), "AAAA 11")


if __name__ == "__main__":
    unittest.main()
