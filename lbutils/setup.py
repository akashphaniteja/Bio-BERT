from setuptools import setup, find_packages
import pkg_resources

version = '1.0.0'

setup(
    name="lbutils",
    version=version,
    description="Helper library for Labelbox NER",
    packages=['lbutils'],
    license="MIT",
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.18.5",
        "pandas>=1.1.5",
    ],
)
