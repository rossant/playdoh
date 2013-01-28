from playdoh import *
from test import *


class UserPrefTest(unittest.TestCase):
    def test_1(self):
        self.assertEqual(type(USERPREF['connectiontrials']), int)
        prevport = USERPREF['port']
        USERPREF['port'] = 123456
        self.assertEqual(USERPREF['port'], 123456)
        USERPREF['port'] = prevport


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(UserPrefTest)


if __name__ == '__main__':
    unittest.main()
