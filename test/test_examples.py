from test import *
import os
import sys
import time
import playdoh


class ExamplesTest(unittest.TestCase):
    def test_1(self):
        examples_dir = '../examples'
        os.chdir(examples_dir)
        skip = ['external_module.py', 'resources.py', 'allocation.py']
        files = os.listdir('.')
        for file in files:
            if file in skip:
                continue
            if os.path.splitext(file)[1] == '.py':
                log_info("Running %s..." % file)
                os.system('python %s' % file)
                time.sleep(.5)


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(ExamplesTest)


if __name__ == '__main__':
    unittest.main()
