'''
NOTE: you need a .pypirc file to do this, you may need to set the
HOME env to where it is saved. Also note that any spaces in the
filename of HOME will cause it to not work, so use old style 8.3
equivalent name.

Also note that manifest.in may not work right with this?
'''

import os

os.chdir('../../.')  # work from project root
os.system('python setup.py register')
# os.system('python setup.py sdist bdist_wininst upload')
