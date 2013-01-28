from playdoh import *
from test import *
from multiprocessing import Process
from threading import Thread
import sys
import time
import unittest
import gc
import cPickle


class ResourcesTest(LocalhostTest):
    def test_1(self):
        cpu = get_total_resources('localhost')[0]['CPU']
        set_total_resources('localhost', CPU=cpu - 1)
        resources = get_total_resources('localhost')[0]
        self.assertEqual(resources['CPU'], cpu - 1)

    def test_2(self):
        request_resources('localhost', CPU=2)
        resources = get_my_resources('localhost')[0]
        self.assertEqual(resources['CPU'], 2)

    def test_optimal(self):
        set_total_resources('localhost', CPU=3)
        request_all_resources('localhost', 'CPU')
        self.assertEqual(get_my_resources('localhost')[0]['CPU'], 3)


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(ResourcesTest)


if __name__ == '__main__':
    unittest.main()
