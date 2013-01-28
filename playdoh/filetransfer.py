"""
Allows to transfer files via RPC. Used for Python modules transmission.
"""
from rpc import *
from debugtools import *
from cache import *
import os.path
import shutil
import base64


def readbinary(filename):
    """
    Converts a binary file into a Python list of chars so that it can
    be pickled without problem.
    """
    binfile = open(filename, 'rb')
    data = binfile.read()
    binfile.close()
    data_base64 = base64.b64encode(data)
    return data_base64


def writebinary(filename, data_base64):
    """
    Writes data stored in datalist (returned by readbinary) into filename.
    """
    binfile = open(filename, 'wb')
    data = base64.b64decode(data_base64)
    binfile.write(data)
    binfile.close()
    return


class FileTransferHandler(object):
#    basedir = os.path.realpath(os.path.dirname(__file__)) # module dir
    basedir = BASEDIR  # user dir

    def save_files(self, files, filenames):
        """
        Saves files on the disk.
        """
        for file, filename in zip(files, filenames):
            # real path to the file on the server
            filename = os.path.realpath(os.path.join(self.basedir, filename))

            # folder on the server
            dirname = os.path.dirname(filename)

            # creates the folder if needed
            if not os.path.exists(dirname):
                try:
                    log_debug("server: creating '%s' folder" % dirname)
                    os.mkdir(dirname)
                except:
                    log_warn("server: error while creating '%s'" % dirname)

            log_debug("server: writing '%s'" % filename)
            writebinary(filename, file)
        return True

    def erase_files(self, filenames=None, dirs=None):
        if filenames is None:
            filenames = []
        if dirs is None:
            dirs = []
        # Deletes filenames
        for filename in filenames:
            filename = os.path.realpath(os.path.join(self.basedir, filename))
            log_debug("server: erasing '%s'" % filename)
            os.remove(filename)
        # Deletes directories
        for dir in dirs:
            dir = os.path.realpath(os.path.join(self.basedir, dir))
            try:
                log_debug("server: erasing '%s' folder" % filename)
                shutil.rmtree(dir)
            except Exception:
                log_warn("server: an error occured when erasing the folder")
        return True


def send_files(machines, from_filenames, to_filenames=None, to_dir=None):
    if type(machines) is str:
        machines = [machines]
    if type(from_filenames) is str:
        from_filenames = [from_filenames]
    if type(to_filenames) is str:
        to_filenames = [to_filenames]
    if len(machines) == 0:
        log_debug("No machines to send files")
        return

    if to_filenames is None:
        if to_dir is None:
            raise Exception("If 'to_filenames' is not specified, you must \
                             specify 'to_dir'")
        else:
            basedir = os.path.commonprefix(from_filenames)
            to_filenames = [os.path.join(to_dir,
                                         os.path.relpath(file, basedir))
                                         for file in from_filenames]
    files = [readbinary(filename) for filename in from_filenames]

    GC.set(machines, handler_class=FileTransferHandler, handler_id="savefiles")
    disconnect = GC.connect()
    result = GC.save_files([files] * len(machines),
                           [to_filenames] * len(machines))
    GC.delete_handler()
    GC.set(machines)  # forget the handler_class/id

    if disconnect:
        GC.disconnect()

#    clients.disconnect()
    return result


def erase_files(machines, filenames=None, dirs=None):
    if type(machines) is str:
        machines = [machines]
    if type(filenames) is str:
        filenames = [filenames]
    if type(dirs) is str:
        dirs = [dirs]

    if filenames is None and dirs is None:
        raise Exception("Both 'filenames' and 'dirs' cannot be None.")

    GC.set(machines, handler_class=FileTransferHandler, handler_id="savefiles")
    disconnect = GC.connect()

#    clients = RpcClients(machines, handler_class=FileTransferHandler,
#               handler_id="savefiles")
#    clients.connect()
#    clients.add_handler([FileTransferHandler]*len(machines))
    result = GC.erase_files([filenames] * len(machines),
                            [dirs] * len(machines))
    GC.delete_handler()

    if disconnect:
        GC.disconnect()

#    clients.disconnect()
    return result
