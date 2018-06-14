"""Setup script for tekel."""
from __future__ import print_function

import sys

from re import search
from setuptools import setup, find_packages
from tekel import __authors__, __version__


def main():
    with open('requirements.txt') as f:
        INSTALL_REQUIRES = [l.strip()
                            for l in f.readlines()
                            if l and not search('# optional', l)]

    setup(name='tekel',
          version=__version__,
          description='A TensorFlow kernel library',
          long_description=open('README.rst').read(),
          license=open('LICENSE').read(),
          authors=__authors__,
          packages=find_packages(),
          install_requires=INSTALL_REQUIRES,
          author_email='mail@romainbrault.com',
          )


if __name__ == '__main__':
    main()
    sys.exit(0)
