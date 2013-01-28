import os


os.chdir('../../.')  # work from project root
os.system('python setup.py bdist_wininst')
os.system('python setup.py sdist --formats=gztar,zip')
