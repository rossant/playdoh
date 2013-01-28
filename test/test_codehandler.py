from playdoh import *
from test import *
from imports.testclass import TestClass
import unittest
import cPickle
import time


class TestHandler(object):
    def test(self, pkl_class):
        o = pkl_class()
        return o.fun(3)


class CodePickleTest(LocalhostTest):
    def test_1(self):
        pkl_class = pickle_class(TestClass)
        send_dependencies('localhost', TestClass)

        client = RpcClient('localhost', handler_class=TestHandler)
        result = client.test(pkl_class)
        [self.assertEqual(result[i], i * i) for i in xrange(2)]


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(CodePickleTest)


if __name__ == '__main__':
    unittest.main()
