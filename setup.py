import os

from setuptools import find_packages, setup

# Recommendations from https://packaging.python.org/
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def read(*parts):
    with open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


about = {}
with open('e3signals/version.py') as f:
    exec(f.read(), about)  # pylint: disable=exec-used

setup(
    name='e3signals',
    version=about['__version__'],
    description='Library that adds functions on top of e3nn to deal with expanding and manipulating signals in a '
                'spherical harmonic basis.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://e3nn.org',
    packages=find_packages(exclude=["tests.*", "tests"]),
    install_requires=[
        'e3nn',
    ],
    extras_require={
        'dev': [
            'pytest',
            'pre-commit',
        ],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
    ],
    python_requires='>=3.7',
    license="MIT",
    license_files="LICENSE",
)
