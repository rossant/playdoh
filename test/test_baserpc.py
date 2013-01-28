from playdoh import *
from test import *
import sys
import time
import unittest


class BaseRpcTest(BaseLocalhostTest):
    def test_1(self):
        client = BaseRpcClient('localhost')
        self.assertEqual(client.execute("hello world"), "hello world")

    def test_2(self):
        client = BaseRpcClient('localhost')
        client.connect()
        self.assertEqual(client.execute("hello world"), "hello world")
        client.disconnect()


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(BaseRpcTest)


if __name__ == '__main__':
    unittest.main()
