from setuptools import setup, find_packages
import pip
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
requirements_file = os.path.join(dir_path, 'requirements.txt')
INSTALL_REQUIRES = [req.name for req in pip.req.parse_requirements(requirements_file, session='hack')]

setup(
    name='spyros',
    version='0.0.1',
    description='Project that centralize a bunch of nan-friendly functions from other packages',
    author='ETS',
    classifiers=[
        'Development Status :: 1 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
		'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(exclude=['tests', 'docs', 'examples']),
    install_requires=INSTALL_REQUIRES,
    zip_safe=True,
)
