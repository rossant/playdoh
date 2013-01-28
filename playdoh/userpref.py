from cache import *
#from debugtools import *
#from .debugtools import level
import os
import os.path
import imp

__all__ = ['DEFAULTPREF', 'USERPREF']

# Default preference values
DEFAULTPREF = {}
# authentication key in the network
DEFAULTPREF['authkey'] = 'playdohauthkey'

# default port
DEFAULTPREF['port'] = 2718

# default port
DEFAULTPREF['connectiontrials'] = 5

# default connectiontimeout (in s)
DEFAULTPREF['connectiontimeout'] = 30

# default port
DEFAULTPREF['favoriteservers'] = []

# default port
DEFAULTPREF['loglevel'] = 'INFO'


class UserPref(object):
    """
    User preferences. Allows to load and save user preference by getting
    the default value if it is not specified by the user, or by loading/saving
    to a file in ``~/.playdoh/userpref.py`` which must defines a dictionary
    named ``USERPREF``.
    To load a user preference, use ``USERPREF[key]`` where ``USERPREF`` is
    a global variable.
    """
    def __init__(self):
        self.preffile = os.path.join(BASEDIR, 'userpref.py')
        self.userpref = {}
        self.load_default()
        self.load()

    def load_default(self):
        """
        Load default values
        """
        for key, val in DEFAULTPREF.iteritems():
            self.userpref[key] = DEFAULTPREF[key]

    def load(self):
        """
        Load values from the user preference file
        """
        if os.path.exists(self.preffile):
            module = imp.load_source('userpref', self.preffile)
            userpref = getattr(module, 'USERPREF')
            for key, val in userpref.iteritems():
                self.userpref[key] = val

    def save(self):
        string = "USERPREF = {}\n"
        for key in self.userpref.keys():
            val = self.userpref[key]
            if type(val) is str or type(val) is unicode:
                string += "USERPREF['%s'] = '%s'\n" % (key, str(val))
            else:
                string += "USERPREF['%s'] = %s\n" % (key, str(val))
        f = open(self.preffile, 'w')
        f.write(string)
        f.close()

    def __getitem__(self, key):
        if key not in self.userpref.keys():
            self.userpref[key] = None
        return self.userpref[key]

    def __setitem__(self, key, val):
        self.userpref[key] = val

USERPREF = UserPref()
