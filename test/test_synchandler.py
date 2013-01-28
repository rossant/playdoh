from playdoh import *
from test import *
from multiprocessing import Process
from numpy.random import rand
from numpy import max
import time


class TaskTest1(ParallelTask):
    def initialize(self, x):
        self.x = x

    def start(self):
        self.result = self.x ** 2

    def get_result(self):
        return self.result


class TaskTest2(TaskTest1):
    def start(self):
#        log_debug(self.tubes)
        self.result = self.x ** 2
        if self.node.index == 0:
#            log_info('I am 0, pushing now')
            self.tubes.push('tube1', 1)
        if self.node.index == 1:
#            log_info('I am 1, popping now')
            self.result += self.tubes.pop('tube1')


class TaskTest4(ParallelTask):
    def initialize(self, x, iterations):
        self.x = x
        self.iterations = iterations

    def start(self):
        for self.iteration in xrange(self.iterations):
#            log_debug('ITERATION %d' % self.iteration)
            self.tubes.push('right', self.x + 1)
            self.x += self.tubes.pop('right')
            time.sleep(.3)
        self.result = self.x

    def get_info(self):
        return self.iteration

    def get_result(self):
        return self.result


class TaskTestShared(ParallelTask):
    def start(self):
        self.result = None
        time.sleep(1)


class SynchandlerTest(unittest.TestCase):
    def test_1(self):
        """
        No tube test
        """
        time.sleep(2)
        x = rand(10)

        allocation = allocate(cpu=1)
        task = start_task(TaskTest1,
                          allocation=allocation,
                          args=(x,))

        result = task.get_result()
        self.assertTrue(max(abs(result - x ** 2)) < 1e-9)

    def test_2(self):
        """
        1 tube on a single machine with 2 CPUs
        """
        time.sleep(2)

        topology = [('tube1', 0, 1)]

        task = start_task(TaskTest2,
                          topology=topology,
                          cpu=2,
                          args=(3,))
        result = task.get_result()
        self.assertEqual(result, [9, 10])

    def test_3(self):
        """
        1 tube on two machines with 1 CPU each
        """
        time.sleep(2)

        p1 = Process(target=open_server, args=(2718, 1, 0))
        p1.start()
        time.sleep(.2)

        p2 = Process(target=open_server, args=(2719, 1, 0))
        p2.start()
        time.sleep(.2)

        machine1 = (LOCAL_IP, 2718)
        machine2 = (LOCAL_IP, 2719)
        machines = [machine1, machine2]
        task_id = "my_task"
        type = 'CPU'

        topology = [('tube1', 0, 1)]

        allocation = allocate(allocation={machine1: 1, machine2: 1},
                                          unit_type=type)

        task = start_task(TaskTest2, task_id, topology,
                          unit_type=type,
                          allocation=allocation,
                          args=(3,))
        result = task.get_result()

        self.assertEqual(result, [9, 10])
        close_servers([machine1, machine2])
        time.sleep(.2)

    def test_4(self):
        """
        4 nodes, 2 machines with 2 CPUs
        """
        time.sleep(2)

        p1 = Process(target=open_server, args=(2718, 2, 0))
        p1.start()
        time.sleep(.2)

        p2 = Process(target=open_server, args=(2719, 2, 0))
        p2.start()
        time.sleep(.2)

        machine1 = (LOCAL_IP, 2718)
        machine2 = (LOCAL_IP, 2719)
        machines = [machine1, machine2]
        task_id = "my_task"
        type = 'CPU'

        topology = [('right', 0, 1),
                    ('right', 1, 2),
                    ('right', 2, 3),
                    ('right', 3, 0)
                    ]

        allocation = allocate(allocation={machine1: 2, machine2: 2},
                                          unit_type=type)
        task = start_task(TaskTest4, task_id, topology,
                          unit_type=type,
                          codedependencies=[],
                          allocation=allocation,
                          args=(0, 3))

        result = task.get_result()
        self.assertEqual(result, [7, 7, 7, 7])

        close_servers([machine1, machine2])
        time.sleep(.2)

    def test_shared(self):
        time.sleep(2)

        topology = []
        x = rand(10)
        shared_data = {'x': x}
        allocation = allocate(cpu=2)

        task = start_task(TaskTestShared,
                          topology=topology,
                          allocation=allocation)
        task.get_result()


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(SynchandlerTest)


if __name__ == '__main__':
    unittest.main()
