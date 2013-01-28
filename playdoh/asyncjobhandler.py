"""
Asynchronous Job Manager
"""
from cache import *
from gputools import *
from pool import *
from rpc import *
from resources import *
from numpy import sum
import cPickle
import os
import os.path
import time
import hashlib
import random
import traceback


__all__ = ['Job', 'JobRun', 'AsyncJobHandler', 'submit_jobs']


class Job(object):
    jobdir = JOBDIR

    def __init__(self, function, *args, **kwds):
        """
        Constructor.

        *Arguments*

        `function`
          The function to evaluate, it is a native Python function. Function
          serialization should take care of ensuring that this function is
          correctly defined in the namespace.

        `*args, **kwds`
          The arguments of the function.
        """
        self.function = function
        self.args = args
        self.kwds = kwds
        self.result = None
        self.status = 'queued'

    def compute_id(self):
        """
        Computes a unique identifier of the job.
        """
        m = hashlib.sha1()
        pid = os.getpid()  # current process id
#        t = int(time.time() * 10000) % 1000000  # time
        s = str(self.function)  # function name
        s += cPickle.dumps(self.args, -1)  # args dump
        s += cPickle.dumps(self.kwds, -1)  # args dump
        m.update(str(random.random()))  # random number
        m.update(str(pid))
        m
        m.update(s)
        hash = m.hexdigest()
        self._id = hash
        return self._id

    def get_id(self):
        if not hasattr(self, '_id'):
            self.compute_id()
        return self._id
    id = property(get_id)

    def get_filename(self):
        return os.path.join(Job.jobdir, self.id + '.pkl')
    filename = property(get_filename)

    def evaluate(self):
        """
        Evaluates the function on the given arguments.
        """
        try:
            self.status = 'processing'
            self.result = self.function(*self.args, **self.kwds)
            self.status = 'finished'
        except Exception as inst:
            # add the traceback to the exception
            msg = traceback.format_exc()
            inst.traceback = msg
            log_warn("An exception has occurred in %s, print exc.traceback \
where exc is the Exception object returned by playdoh.map" %
                self.function.__name__)
            self.result = inst
            self.status = 'crashed'
        return self.result

    def record(self):
        """
        Records the job after evaluation on the disk.
        """
        if not os.path.exists(self.jobdir):
            log_debug("creating '%s' folder for code pickling" % self.jobdir)
            os.mkdir(self.jobdir)
        log_debug("writing '%s'" % self.filename)

        # delete shared data before pickling
        if 'shared_data' in self.kwds:
            del self.kwds['shared_data']

        file = open(self.filename, 'wb')
        cPickle.dump(self, file, -1)
        file.close()

    @staticmethod
    def load(id):
        """
        Returns the Job object stored in the filesystem using its identifier.
        """
        try:
            filename = os.path.join(Job.jobdir, id + '.pkl')
            log_debug("opening file '%s'" % filename)
            file = open(filename, 'rb')
            job = cPickle.load(file)
            file.close()
#            time.sleep(.005)
        except IOError:
            log_debug("file '%s' not found" % filename)
            job = None
        except EOFError:
            log_debug("EOF error with '%s', trying again..." % filename)
            time.sleep(.2)
            file = open(filename, 'rb')
            job = cPickle.load(file)
            file.close()
        return job

    @staticmethod
    def erase(id):
        """
        Erases the Job object stored in the filesystem using its identifier.
        """
        filename = os.path.join(Job.jobdir, id + '.pkl')
        log_debug("erasing '%s'" % filename)
        try:
            os.remove(filename)
        except:
            log_warn("Unable to erase <%s>" % filename)

    @staticmethod
    def erase_all():
        """
        Erases all Job objects stored in the filesystem.
        """
        files = os.listdir(Job.jobdir)
        log_debug("erasing all files in '%s'" % Job.jobdir)
        [os.remove(os.path.join(Job.jobdir, filename)) for filename in files]


def eval_job(job, shared_data={}):
    """
    Evaluates a job. Must be global to be passed to CustomPool.
    Handles Exceptions.
    """
    if len(shared_data) > 0:
        job.kwds['shared_data'] = shared_data
    result = job.evaluate()
    job.record()
    return result


class AsyncJobHandler(object):
    """
    A Handler object handling asynchronous job management, on the server side.
    """
    def __init__(self):
        """
        max_cpu is the maximum number of CPUs dedicated to the cluster
        idem for max_gpu
        None = use all CPUs/GPUs available
        """
        self.handlers = []
        self.cpu = MAXCPU
        self.gpu = 0
        self.pool = None
        self.cpool = None
        self.jobs = {}

    def add_jobs(self, jobs):
        for job in jobs:
            self.jobs[job.id] = job
        return [job.id for job in jobs]

    def initialize_cpool(self, type, units, do_redirect):
        pool = self.pools[type]
#        status = pool.get_status()
        unitindices = pool.get_idle_units(units)
        if len(unitindices) != units:
            msg = "not enough %s(s) available, exiting now" % (type)
            log_warn(msg)
            raise Exception(msg)
        log_debug("found %d %s(s) available: %s" % (units, type,
                                                    str(unitindices)))

        self.cpool = CustomPool(unitindices, unittype=type,
                                do_redirect=do_redirect)
        # links the global Pool object to the CustomPool object
        self.cpool.pool = pool

    def submit(self, jobs, type='CPU', units=None, shared_data={},
               do_redirect=None):
        """
        Submit jobs.

        *Arguments*

        `jobs`
          A list of Job objects.
        """
        job_ids = self.add_jobs(jobs)
        # By default, use all resources assigned to the current client
        # for this handler.
        # If units is set, then use only this number of units
#        if units is None:
#            units = self.resources[type][self.client]

        # find idle units
        if units is None:
            log_warn("units should not be None in submit")

        if self.cpool is None:
            self.initialize_cpool(type, units, do_redirect)
        else:
            self.cpool.set_units(units)

        pool_ids = self.cpool.submit_tasks(eval_job, shared_data, jobs)
        for i in xrange(len(jobs)):
            id = job_ids[i]
            self.jobs[id].pool_id = pool_ids[i]

        return job_ids

    def get_pool_ids(self, job_ids):
        """
        Converts job ids (specific to AsyncJobHander) to pool ids
        (specific to the CustomPool object)
        """
        return [self.jobs[id].pool_id for id in job_ids]

    def get_status(self, job_ids):
        if job_ids is None:
            statuss = None
            raise Exception("The job identifiers must be specified")
        else:
            statuss = []
            for id in job_ids:
                job = Job.load(id)
                if job is not None:
                    log_debug("job file '%s' found" % id)
                    status = job.status
                elif id in self.jobs.keys():
                    log_debug("job file '%s' not found" % id)
                    status = self.jobs[id].status
                else:
                    log_warn("job '%s' not found" % id)
                    status = None
                statuss.append(status)
        return statuss

    def get_results(self, job_ids):
        if job_ids is None:
            results = None
            raise Exception("Please specify job identifiers.")
        else:
            results = []
            for id in job_ids:
                job = Job.load(id)
                if job is not None:
                    result = job.result
                else:
                    # if job is None, it means that it probably has
                    # not finished yet
                    result = None
#                    if self.pool is not None:
                    log_debug("Tasks have not finished yet, waiting...")
                    self.cpool.join()
                    job = Job.load(id)
                    if job is not None:
                        result = job.result
                results.append(result)
        return results

    def has_finished(self, job_ids):
        if self.cpool is not None:
            pool_ids = self.get_pool_ids(job_ids)
            return self.cpool.has_finished(pool_ids)
        else:
            log_warn("The specified job identifiers haven't been found")
            return None

    def erase(self, job_ids):
        log_debug("Erasing job results")
        [Job.erase(id) for id in job_ids]

    def close(self):
        if hasattr(self, 'cpool'):
            if self.cpool is not None:
                self.cpool.close()
            else:
                log_warn("The pool object has already been closed")

    def kill(self):
        # TODO: jobids?
        if self.cpool is not None:
            self.cpool.kill()
        else:
            log_warn("The pool object has already been killed")


class JobRun(object):
    """
    Contains information about a parallel map that has been launched
    by the ``map_async`` function.

    Methods:

    ``get_status()``
        Returns the current status of the jobs.

    ``get_result(jobids=None)``
        Returns the result. Blocks until the jobs have finished.
        You can specify jobids to retrieve only some of the results,
        in that case it must
        be a list of job identifiers.
    """
    def __init__(self, type, jobs, machines=[]):
        self.type = type
        self.jobs = jobs
        self.machines = machines  # list of Machine object
        self._machines = [m.to_tuple() for m in self.machines]
        self.local = None
        self.jobids = None

    def set_local(self, v):
        self.local = v

    def set_jobids(self, jobids):
        self.jobids = jobids

    def get_machines(self):
        return self._machines

    def get_machine_index(self, machine):
        for i in xrange(len(self.machines)):
            if (self.machines[i] == machine):
                return i

    def concatenate(self, lists):
        lists2 = []
        [lists2.extend(l) for l in lists]
        return lists2

    def get_status(self):
        GC.set(self.get_machines(), handler_class=AsyncJobHandler)
        disconnect = GC.connect()

        status = GC.get_status(self.jobids)

        if disconnect:
            GC.disconnect()

        return self.concatenate(status)

    def get_results(self, ids=None):
        if ids is None:
            ids = self.jobids
        GC.set(self.get_machines(), handler_class=AsyncJobHandler)
        disconnect = GC.connect()

        if not self.local:
            log_info("Retrieving job results...")
        results = GC.get_results(ids)
        GC.erase(self.jobids)
        if disconnect:
            GC.disconnect()

#        clients = RpcClients(self.get_machines(),
#            handler_class=AsyncJobHandler)
#        clients.connect()
#        results = clients.get_results(self.jobids)
#        clients.erase(self.jobids)
#        clients.disconnect()

        results = self.concatenate(results)
        if self.local:
            close_servers(self.get_machines())
        return results

    def get_result(self):
        return self.get_results()

    def __repr__(self):
        nmachines = len(self.machines)
        if nmachines > 1:
            plural = 's'
        else:
            plural = ''
        return "<Task: %d jobs on %d machine%s>" % (len(self.jobs),
                                                    nmachines, plural)


def create_jobs(fun, argss, kwdss):
    """
    Create Job objects
    """
    jobs = []
    k = len(argss)  # number of non-named arguments
    keys = kwdss.keys()  # keyword arguments

    i = 0  # task index
    while True:
        try:
            args = [argss[l][i] for l in xrange(k)]
            kwds = dict([(key, kwdss[key][i]) for key in keys])
        except:
            break
        jobs.append(Job(fun, *args, **kwds))
        i += 1
    return jobs


def split_jobs(jobs, machines, allocation):
    """
    Splits jobs among workers
    """
    total_units = allocation.total_units
    njobs = len(jobs)

    # charge[i] is the number of jobs on machine #i
    i = 0  # worker index
    charge = []
    for m in machines:
        nbr_units = allocation[m]  # number of workers on this machine
        charge.append(nbr_units * njobs / total_units)
        i += 1
    charge[-1] = njobs - sum(charge[:-1], dtype=int)

    sjobs = []
    i = 0  # worker index
    total = 0  # total jobs
    for m in machines:
        k = charge[i]
        sjobs.append(jobs[total:(total + k)])
        total += k
        i += 1
        if total >= njobs:
            break
    return sjobs


def submit_jobs(fun,
                allocation,
                unit_type='CPU',
                shared_data={},
                local=None,
                do_redirect=None,
                argss=[],
                kwdss={}):
    """
    Submit map jobs. Use ``map_async`` instead.
    """
    machines = allocation.machines

    # creates Job objects
    jobs = create_jobs(fun, argss, kwdss)

    # creates a JobRun object
    myjobs = JobRun(unit_type, jobs, machines)

    # splits jobs
    sjobs = split_jobs(jobs, machines, allocation)
    units = [allocation[m] for m in myjobs.get_machines()]

    # are jobs running locally?
    if local is None and (len(machines) == 1) and (machines[0].ip == LOCAL_IP):
        myjobs.set_local(True)
    if local is not None:
        myjobs.set_local(local)

    GC.set(myjobs.get_machines(), handler_class=AsyncJobHandler)
    disconnect = GC.connect()

    # Submits jobs to the machines
#    clients = RpcClients(myjobs.get_machines(), handler_class=AsyncJobHandler)

    jobids = GC.submit(sjobs, type=unit_type, units=units,
                       shared_data=shared_data,
                       do_redirect=do_redirect)

    if disconnect:
        GC.disconnect()

    # Records job ids
    myjobs.set_jobids(jobids)

    return myjobs
