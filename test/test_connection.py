from playdoh import *
from test import *
from multiprocessing import Process
from threading import Thread
import sys
import time
import unittest
import gc
import cPickle
from numpy.random import rand


def serve():
    # echo server
    conn, client = accept(('', 2718))
    while True:
        data = conn.recv()
        if data is None:
            break
        log_debug("received %d" % len(data))
        conn.send(data)
    conn.close()


class ConnectionTest(unittest.TestCase):
    def test_1(self):
        p = Process(target=serve)
        p.start()

        conn = connect(('localhost', 2718))
        for i in xrange(20):
            data = cPickle.dumps(rand(100, 100) * 100000)
            log_debug("%d, sending %d" % (i, len(data)))
            conn.send(data)
            data2 = conn.recv()
            self.assertEqual(data, data2)
        conn.send(None)
        conn.close()
        p.terminate()
        p.join()


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(ConnectionTest)


if __name__ == '__main__':
    unittest.main()
