import os.path

# creates the user directory
BASEDIR = os.path.join(os.path.realpath(os.path.expanduser('~')), '.playdoh')
CACHEDIR = os.path.join(BASEDIR, 'cache')
JOBDIR = os.path.join(BASEDIR, 'jobs')

if not os.path.exists(BASEDIR):
    os.mkdir(BASEDIR)

if not os.path.exists(CACHEDIR):
    os.mkdir(CACHEDIR)

if not os.path.exists(JOBDIR):
    os.mkdir(JOBDIR)
