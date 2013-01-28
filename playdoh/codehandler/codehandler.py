from ..rpc import *
from ..filetransfer import *
from ..cache import *
from ..debugtools import *
import inspect
import os
import os.path
import sys
import cPickle
import imp
import hashlib
from cloudpickle import dumps, dump


SYSPATH = sys.path[:]


__all__ = ['PicklableClass', 'pickle_class', 'send_dependencies', 'dump']


class PicklableClass(object):
    """
    Makes a class picklable and allows the creation of an instance of
    this class
    on a remote computer. The source code of the class is saved in a string
    and recorded at runtime on the file system before being dynamically
    imported.
    This basic class doesn't handle dependencies with external files.
    """
    codedir = CACHEDIR

    def __init__(self, myclass):
        self.__name__ = self.classname = myclass.__name__
        self.pkl = dumps(myclass)
        try:
            self.source = inspect.getsource(myclass).strip()
        except:
            log_debug("could not get the source code of the function or \
                class <%s>" % self.__name__)
            self.source = ""
        sha = hashlib.sha1()
        sha.update(self.source)
#        sha.update(str(random.random())) # HACK to avoid conflicts between
#        local and remote scripts
        self.hash = sha.hexdigest()
        self.isfunction = inspect.isfunction(myclass)
        if self.isfunction:
            self.arglist = inspect.getargspec(myclass)[0]
        self.distant_dir = os.path.join(self.codedir, self.hash)
        self.codedependencies = []
#        self.isloaded = False

    def set_code_dependencies(self, codedependencies=[]):
        self.codedependencies = codedependencies

    def load_pkl(self):
        global SYSPATH
        sys.path = SYSPATH[:]
        log_debug("Adding directory <%s> to sys.path" % self.distant_dir)
        sys.path.append(self.distant_dir)
#        log_debug(sys.path)
#        self.isloaded = True

    def load_modules(self):
        """
        Loads the dependencies
        """
        loadedmodules = {}
        for m in self.codedependencies:
            name = m
            name = name.replace('.py', '')
            name = name.replace('//', '.')
            name = name.replace('/', '.')
            name = name.replace('\\\\', '.')
            name = name.replace('\\', '.')
            loadedmodules[name] = imp.load_source(name,
                                            os.path.join(self.distant_dir, m))
        return loadedmodules

    def __call__(self, *args, **kwds):
        """
        Instantiates an object of the class.
        """
#        if not self.isloaded:
        self.load_pkl()
        self.load_modules()
        myclass = cPickle.loads(self.pkl)
        return myclass(*args, **kwds)


def get_dirname(myclass):
    modulename = myclass.__module__
    module = sys.modules[modulename]
    if hasattr(module, '__file__'):
        # path of the file where the class is defined
        path = os.path.realpath(module.__file__)
    else:
        # otherwise : current path
        path = os.path.realpath(os.getcwd())
    dirname, filename = os.path.split(path)
    return dirname


def get_dependencies(myclass, dirname=None):
    if dirname is None:
        dirname = get_dirname(myclass)
    filelist = []
    for dir, dirnames, filenames in os.walk(dirname):
        for name in filenames:
            filename = os.path.realpath(name)
            ext = os.path.splitext(filename)[1]
            if ext == '.py':
                filelist.append(os.path.join(dir, name))
    return filelist


def pickle_class(myclass):
    return PicklableClass(myclass)


def send_dependencies(servers, myclass, dependencies=[]):
    dirname = get_dirname(myclass)
    pklclass = pickle_class(myclass)
#    if dependencies is None:
#        dependencies = get_dependencies(myclass, dirname)
    if dependencies is None:
        dependencies = []
    dep = []
    if len(dependencies) > 0:
        to_filenames = [os.path.join(pklclass.distant_dir,
                                     os.path.relpath(filename, dirname))\
                                     for filename in dependencies]
        dep = [os.path.relpath(filename, dirname) for filename in dependencies]
        log_info("Sending code dependencies (%d file(s))" % len(to_filenames))
        send_files(servers, dependencies, to_filenames)
    pklclass.set_code_dependencies(dep)
    return pklclass
