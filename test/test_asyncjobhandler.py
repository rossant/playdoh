from playdoh import *
from test import *
from multiprocessing import Process
import sys
import time
import unittest


def fun(x):
    return x * x


class ASyncJobHandlerTest(LocalhostTest):
    def test_1(self):
        # new connection to localhost
        client = RpcClient('localhost', handler_class=AsyncJobHandler)
        client.connect()
        jobs = [Job(fun, i) for i in xrange(6)]

        jobids = client.submit(jobs, units=2)

        time.sleep(2)
        status = client.get_status(jobids)
        for s in status:
            self.assertEqual(s, 'finished')

        results = client.get_results(jobids)
        client.disconnect()

        for i in xrange(len(results)):
            self.assertEqual(results[i], i * i)

        # erases the jobs
        [Job.erase(id) for id in jobids]


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(ASyncJobHandlerTest)


if __name__ == '__main__':
    unittest.main()
