import unittest
import os
import os.path
import re
import time
import multiprocessing
from playdoh import *


class BaseLocalhostTest(unittest.TestCase):
    def setUp(self):
        log_debug("STARTING SERVER FOR TEST %str" % self.id())
        self.p = multiprocessing.Process(target=open_base_server)
        self.p.start()
        time.sleep(.1)

    def tearDown(self):
        log_debug("CLOSING SERVER")
        close_base_servers('localhost')
        self.p.join(3.0)
        time.sleep(.2)


class LocalhostTest(unittest.TestCase):
    def setUp(self):
        if hasattr(self, 'maxcpu'):
            maxcpu = self.maxcpu
        else:
            maxcpu = None
        if hasattr(self, 'maxgpu'):
            maxgpu = self.maxgpu
        else:
            maxgpu = None

        log_debug("STARTING SERVER FOR TEST %str" % self.id())
        self.p = multiprocessing.Process(target=open_server,
                                         args=(None, maxcpu, maxgpu))
        self.p.start()
        time.sleep(.1)

    def tearDown(self):
        log_debug("CLOSING SERVER")
        close_servers('localhost')
        self.p.join(3.0)
        time.sleep(.2)


def all_tests(folder=None):
    if folder is None:
        folder = os.path.dirname(os.path.realpath(__file__))
    pattern = '^(test_[^.]+).py$'
    files = os.listdir(folder)
    files = [file for file in files if re.match(pattern, file)]

    suites = []
    for file in files:
        if file in skip:
            continue
        modulename = re.sub(pattern, '\\1', file)
        module = __import__(modulename)
        try:
            suites.append(module.test_suite())
        except:
            log_warn("module '%s' has no method 'test_suite'" % modulename)

    allsuites = unittest.TestSuite(suites)
    return allsuites


def test():
    unittest.main(defaultTest='all_tests')


skip = ['test_examples.py']


if __name__ == '__main__':
    test()
