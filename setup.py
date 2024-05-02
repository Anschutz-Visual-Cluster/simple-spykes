from setuptools import setup, find_packages

VERSION = "0.1.0"


def parse_requirements(requirement_file):
    with open(requirement_file) as fi:
        return fi.readlines()


with open('./README.rst') as f:
    long_description = f.read()


setup(
    name='simple_spykes',
    packages=find_packages(),
    version=VERSION,
    description='Utility library for spike sorting quality metrics',
    author='Spencer Hanson',
    long_description=long_description,
    install_requires=parse_requirements('requirements.txt'),
    keywords=['neuroscience', 'spike sorting', 'tools', 'science'],
    classifiers=[
        'Programming Language :: Python :: 3'
    ]
)

