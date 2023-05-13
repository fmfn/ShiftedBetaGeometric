from setuptools import setup, find_packages

install_requires = [
    "numpy>=1.10.0",
    "scipy>=1.7.1",
    "pandas>=1.3.4",
]

setup(
    name="axia",
    version="0.1.0",
    description=(),
    classifiers=["Programming Language :: Python :: 3"],
    author="Fernando Nogueira",
    author_email="fmfnogueira@gmail.com",
    url="",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
)
