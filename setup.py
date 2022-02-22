from setuptools import setup, find_packages

setup(
    name="mcdc_tnt",
    version="0.1",
    packages=find_packages(include=["mcdc_tnt", "mcdc_tnt.*"]),
    include_package_data=True
)
