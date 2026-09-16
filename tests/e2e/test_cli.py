"""End-to-end integration and CLI workflow tests."""
import subprocess
import sys
import unittest
from pathlib import Path
import pytest


class TestCLIWorkflow(unittest.TestCase):
    def test_cli_help(self):
        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("AdvancEye 2.0 Biometric Attendance System", result.stdout)
        self.assertIn("register", result.stdout)
        self.assertIn("run", result.stdout)
        self.assertIn("list", result.stdout)
        self.assertIn("attendance", result.stdout)

    def test_cli_list_empty(self):
        result = subprocess.run(
            [sys.executable, "main.py", "list"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Registered Student Profiles", result.stdout)


if __name__ == "__main__":
    unittest.main()
