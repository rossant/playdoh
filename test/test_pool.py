from playdoh import *
import time
import unittest
import sys
from numpy import arange, max
from numpy.random import rand


def fun(x, y, wait=.1):
    time.sleep(wait)
    return x + y


def test_numpy(index, child_conns, parent_conns, x):
    conn = child_conns[index]
    time.sleep(1)
    conn.send(x)


def serve(index, child_conns, parent_conns):
    log_info("Serving started")
    while True:
        str = child_conns[index].recv()
        if str is None:
            break
        log_info("Hello %s" % str)
    log_info("Serving finished")


class PoolTest(unittest.TestCase):
    def test_1(self):
        p = CustomPool(2)
        results = p.map(fun, {}, [1, 2], [3, 4])
        self.assertEqual(results[0], 4)
        self.assertEqual(results[1], 6)

        p.clear_tasks()

        ids = p.submit_tasks(fun, {}, [1, 2], [3, 4])
        self.assertEqual(ids[1], 1)  # checks that the tasks have been cleared
        self.assertEqual(p.get_results(ids)[1], 6)

        p.close()

    def test_2(self):
        p = CustomPool(2)
        ilist = arange(10)

        ids1 = p.submit_tasks(fun, {}, x=ilist, y=10 * ilist)
        time.sleep(.5)
        ids2 = p.submit_tasks(fun, {}, x=ilist, y=10 * ilist)

        for _ in xrange(5):
            if p.has_finished(ids2):
                break
            log_debug("waiting")
            time.sleep(.5)
        self.assertEqual(p.has_finished(ids2), True)

        results1 = p.get_results(ids1)
        results2 = p.get_results(ids2)

        p.close()

        for i in ilist:
            self.assertEqual(results1[i], results2[i])
            self.assertEqual(results1[i], 11 * i)

    def test_status(self):
        pool = Pool(3)
        # 3 CPUs

        p = CustomPool([1, 2])
        # Using only CPU #1 and #2
        p.pool = pool

        ilist = arange(2)

        ids1 = p.submit_tasks(fun, {}, x=ilist, y=10 * ilist,
                                    wait=[1] * len(ilist))
        time.sleep(.1)
        self.assertEqual(p.pool.get_status(), ['idle', 'busy', 'busy'])

        results1 = p.get_results(ids1)
        time.sleep(.1)
        self.assertEqual(p.pool.get_status(), ['idle'] * 3)

        p.close()

        for i in ilist:
            self.assertEqual(results1[i], 11 * i)

    def test_pools(self):
        """
        This test illustrates how a single pool of processes
        (as many processes as CPUs)
        can be used with multiple CustomPool objects.
        """
        pool = CustomPool.create_pool(4)

        cp1 = CustomPool([0])
        cp1.pool = pool

        cp2 = CustomPool([1, 2, 3])
        # the two CustomPool objects refer to the same Pool object.
        cp2.pool = pool

        self.assertEqual(pool.get_status(), ['idle'] * 4)

        ids1 = cp1.submit_tasks(fun, {}, [1, 2], [3, 4], [1] * 4)
        time.sleep(.1)
        self.assertEqual(pool.get_status(), ['busy', 'idle', 'idle', 'idle'])

        ids2 = cp2.submit_tasks(fun, {}, [1, 2, 3], [5, 6, 7], [1] * 3)
        time.sleep(.1)
        self.assertEqual(pool.get_status(), ['busy', 'busy', 'busy', 'busy'])

        r1 = cp1.get_results(ids1)
        r2 = cp2.get_results(ids2)

        time.sleep(.1)
        self.assertEqual(pool.get_status(), ['idle'] * 4)

        self.assertEqual(r1[1], 6)
        self.assertEqual(r2[-1], 10)

        pool.close()


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(PoolTest)


if __name__ == '__main__':
    unittest.main()
