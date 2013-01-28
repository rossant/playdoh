from playdoh import *
from test import *
from multiprocessing import Process
import sys
import time
import unittest


def fun(x):
    return x * x


class MapTest(LocalhostTest):
    def test_1(self):
        r = map(fun, [1, 2, 3, 4], machines='localhost')
        self.assertEqual(r, [1, 4, 9, 16])
        set_total_resources(['localhost'], CPU=2)
        r = map(fun, [1, 2, 3, 4], machines='localhost')
        self.assertEqual(r, [1, 4, 9, 16])


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(MapTest)


if __name__ == '__main__':
    unittest.main()
