import ez_setup
ez_setup.use_setuptools()

from setuptools import *
#from distutils.core import setup

version = '0.3.1'

long_description = '''Playdoh is a pure Python library for distributing
computations across the free computing units (CPUs and GPUs) available
in a small network of multicore computers. Playdoh supports independent
(embarassingly) parallel problems as well as loosely coupled tasks such
as global optimizations, Monte Carlo simulations and numerical integration
of partial differential equations. It is designed to be lightweight and
easy to use and should be of interest to scientists wanting to turn their
lab computers into a small cluster at no cost.'''

if __name__ == "__main__":
    setup(name='playdoh',
      version=version,
      packages=['playdoh',
                  'playdoh.codehandler',
                  'playdoh.optimization',
                  'playdoh.scripts',
                  ],
      entry_points={
        'console_scripts': [
            'playdoh = playdoh.scripts:run_console',
        ],
        'gui_scripts': [
            'playdoh_gui = playdoh.scripts:gui'],
      },
      install_requires=['numpy>=1.3.0',
                        'scipy',
                        ],
      url='http://code.google.com/p/playdoh/',
      description='Open-source library for distributing computations \
over multiple cores and machines',
      long_description=long_description,
      author='Cyrille Rossant, Bertrand Fontaine, Dan F. M. Goodman',
      author_email='Cyrille.Rossant at ens.fr',
      download_url='http://code.google.com/p/playdoh/downloads/list',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering'
        ]
      )
