"""Setup configuration for solver-error-classifier module."""

import os
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand  # noqa

try:
    from setuptools import find_namespace_packages
except ImportError:
    # A dirty workaround for older setuptools.
    def find_namespace_packages(path="thoth"):
        """Find namespace packages alternative."""
        packages = set()
        for dir_name, dir_names, file_names in os.walk(path):
            if os.path.basename(dir_name) != "__pycache__":
                packages.add(dir_name.replace("/", "."))

        return sorted(packages)


def get_install_requires():
    """Get requirements for solver-error-classifier module."""
    with open("requirements.txt", "r") as requirements_file:
        res = requirements_file.readlines()
        return [req.split(" ", maxsplit=1)[0] for req in res if req]


def get_version():
    """Get current version of solver-error-classifier module."""
    with open(os.path.join("template", "version.py")) as f:
        content = f.readlines()

    for line in content:
        if line.startswith("__version__ ="):
            # dirty, remove trailing and leading chars
            return line.split(" = ")[1][1:-2]
    raise ValueError("No version identifier found")


def read(fname):
    """Read."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


class Test(TestCommand):
    """Introduce test command to run testsuite using pytest."""

    _IMPLICIT_PYTEST_ARGS = [
        "--timeout=120",
        "--cov=./thoth",
        "--mypy",
        "--capture=no",
        "--verbose",
        "-l",
        "-s",
        "-vv",
        "--hypothesis-show-statistics",
        "tests/",
    ]

    user_options = [("pytest-args=", "a", "Arguments to pass into py.test")]

    def initialize_options(self):
        """Initialize cli options."""
        super().initialize_options()
        self.pytest_args = None

    def finalize_options(self):
        """Finalize cli options."""
        super().finalize_options()
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        """Run module tests."""
        import pytest

        passed_args = list(self._IMPLICIT_PYTEST_ARGS)

        if self.pytest_args:
            self.pytest_args = [arg for arg in self.pytest_args.split() if arg]
            passed_args.extend(self.pytest_args)

        sys.exit(pytest.main(passed_args))


VERSION = get_version()
setup(
    name="thoth-solver-error-classifier",
    version=VERSION,
    description="Solver error log classifier for the Thoth project",
    long_description=read("README.md"),
    author="Bjoern Hasemann",
    author_email="bhaseman@redhat.com",
    license="GPLv3+",
    packages=find_namespace_packages(),
    url="https://github.com/thoth-station/solver-error-classfier",
    download_url="https://pypi.org/project/solver-error-classifier",
    package_data={"thoth.solver-error-classifier": ["py.typed", "data/tensorflow/api.json"]},
    entry_points={"console_scripts": ["thoth-solver-error-classifier=thoth.solver.error.classifier.cli:cli"]},
    zip_safe=False,
    install_requires=get_install_requires(),
    cmdclass={"test": Test},
    long_description_content_type="text/x-rst",
    command_options={
        "build_sphinx": {
            "version": ("setup.py", VERSION),
            "release": ("setup.py", VERSION),
        }
    },
)
