"""Setup for pip package."""

from setuptools import find_namespace_packages
from setuptools import setup


def _get_booba_version():
  with open('booba/__init__.py') as fp:
    for line in fp:
      if line.startswith('__version__'):
        g = {}
        exec(line, g)
        return g['__version__']
    raise ValueError('`__version__` not defined in `booba/__init__.py`')


def _parse_requirements(requirements_txt_path):
  with open(requirements_txt_path) as fp:
    return fp.read().splitlines()


_VERSION = _get_sonnet_version()


from setuptools import setup

setup(
    name='booba-libdnn',
    version='_VERSION',
    packages=find_namespace_packages(exclude=['*_test.py']),
    url='https://github.com/amonteir/libdeeplearning',
    license='GNU GENERAL PUBLIC LICENSE',
    author='angelogg',
    author_email='afrmonteiro@gmail.com',
    description=(
        'Booba is a library for building deep neural networks.'),
    requires_python='>=3.6',
    include_package_data=True,
    zip_safe=False,
    install_requires=_parse_requirements('requirements.txt'),
)
