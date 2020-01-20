"""
Build and install the package.
"""
from os import path

from setuptools import find_packages, setup

NAME = "pykinship"
FULLNAME = "Facial Recognition Bias and the BFW Dataset"
AUTHOR = "Joseph Robinson"
AUTHOR_EMAIL = "robinson.jo@northeastern.edu"
LICENSE = "MIT License"
URL = "https://github.com/visionjo/pykinship"
DESCRIPTION = "Automatic kinship recognition"
KEYWORDS = """ 
        kinship faces recognition biometrics machinelearning deeplearning 
        computervision evaluation benchmark data imageset"""

dir_root = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(dir_root, "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

VERSION = "0.1.0"

PACKAGES = find_packages(exclude=["tests", "notebooks", "experiments"])
SCRIPTS = []

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Topic :: Scientific/Engineering",
    "Topic :: Software Development :: Libraries",
    "Programming Language :: Python :: 3.7",
    "License :: OSI Approved :: {}".format(LICENSE),
]
PLATFORMS = ["osx", 'linux']
INSTALL_REQUIRES = []

if __name__ == "__main__":
    setup(
        name=NAME,
        fullname=FULLNAME,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        license=LICENSE,
        url=URL,
        platforms=PLATFORMS,
        scripts=SCRIPTS,
        packages=PACKAGES,
        classifiers=CLASSIFIERS,
        keywords=KEYWORDS,
        install_requires=INSTALL_REQUIRES,
    )



