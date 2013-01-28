from playdoh import *
from test import *
from multiprocessing import Process
from threading import Thread
import sys
import time


class TestHandler(object):
    def __init__(self):
        self.x = 0

    def set(self, x):
        self.x = x

    def test(self, y):
        return self.x + y

    def test1(self):
        log_info("ONE: %d" % self.x)
        time.sleep(1)
        log_info("ONE AGAIN: %d" % self.x)
        return self.x

    def test2(self):
        log_info("TWO")
        self.x = 10


class RpcTest(LocalhostTest):
    def atest_1(self):
        client = RpcClient('localhost', handler_class=TestHandler)
        client.connect()
        client.set(1)
        self.assertEqual(client.test(2), 3)
        client.set_handler_id('new_handler')  # creates a new handler
        self.assertEqual(client.test(3), 3)
        client.disconnect()

    def test_clients(self):
        servers = [('127.0.0.10', 2719), ('127.0.0.20', 2720)]

        p1 = Process(target=open_server, args=(servers[0][1], 1))
        p1.start()

        p2 = Process(target=open_server, args=(servers[1][1], 1))
        p2.start()

        clients = RpcClients(servers, handler_class=TestHandler)
        clients.connect()
        clients.set([10, 20])
        results = clients.test([1, 2])
        self.assertEqual(results[0], 11)
        self.assertEqual(results[1], 22)

        clients.set_client_indices([0])
        clients.set([100])
        results = clients.test([1])
        self.assertEqual(results[0], 101)

        clients.set_client_indices([1])
        clients.set([200])
        clients.set_client_indices(None)
        results = clients.test([1, 1])
        self.assertEqual(results[0], 101)
        self.assertEqual(results[1], 201)

        clients.disconnect()
        close_servers(servers)

        p1.join()
        p2.join()
        p1.terminate()
        p2.terminate()

    def _concurrent1(self):
        client = RpcClient('localhost', handler_class=TestHandler)
        self.r1 = client.test1()

    def _concurrent2(self):
        client = RpcClient('localhost', handler_class=TestHandler)
        self.r2 = client.test2()

    def atest_concurrent(self):
        """
        Two clients simultaneously connect to the same server, the
        two connections are simultaneously active.
        """
        # the first thread displays a variable on the distant handler
        # twice, within a 1 second interval
        t1 = Thread(target=self._concurrent1)
        t1.start()

        time.sleep(.05)

        # the second thread changes the value of the variable during
        # that second
        t2 = Thread(target=self._concurrent2)
        t2.start()

        t1.join()
        t2.join()

        self.assertEqual(self.r1, 10)


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(RpcTest)


if __name__ == '__main__':
    unittest.main()
