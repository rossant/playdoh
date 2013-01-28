from playdoh import *
from test import *
from multiprocessing import Process
import sys
import time
import unittest
import gc


class FileTransferTest(LocalhostTest):
    def test_1(self):
        filename = os.path.realpath(__file__)
        result = send_files('localhost', filename, 'cache/TEST_codepickle.py')
        self.assertEqual(result[0], True)

        result = erase_files('localhost', 'cache/TEST_codepickle.py')
        self.assertEqual(result[0], True)


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(FileTransferTest)


if __name__ == '__main__':
    unittest.main()
