import unittest
import axelrod_dojo as dojo


class TestVersion(unittest.TestCase):
    def test_version_exists(self):
        self.assertIsInstance(dojo.__version__, str)
