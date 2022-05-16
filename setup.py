import os

from pybind11 import get_cmake_dir  # noqa: F401
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join(".", "requirements.txt")
    with open(reqs_path, "r") as f:
        requirements = [line.rstrip() for line in f]
    return requirements


__version__ = "0.0.0"
ext_modules = [
    Pybind11Extension(
        "aijack_dp_core",
        ["src/aijack/defense/dp/core/main.cpp"],
        define_macros=[("VERSION_INFO", __version__)],
    ),
    Pybind11Extension(
        "aijack_secureboost",
        ["src/aijack/collaborative/secureboost/core/main.cpp"],
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

console_scripts = []

setup(
    name="aijack",
    version=__version__,
    description="package to implemet attack and defense method for machine learning",
    author="Hideaki Takahashi",
    author_email="koukyosyumei@hotmail.com",
    license="MIT",
    install_requires=read_requirements(),
    url="https://github.com/Koukyosyumei/AIJack",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    entry_points={"console_scripts": console_scripts},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
