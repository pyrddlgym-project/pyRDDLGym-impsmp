# This file is part of pyRDDLGym-symbolic.
#
# pyRDDLGym-symbolic is free software: you can redistribute it and/or modify
# it under the terms of the MIT License as published by
# the Free Software Foundation.
#
# pyRDDLGym-symbolic is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# MIT License for more details.
#
# You should have received a copy of the MIT License
# along with pyRDDLGym-symbolic. If not, see <https://opensource.org/licenses/MIT>.

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
      name='pyRDDLGym-impsmp',
      version='0.1',
      author="Ilia Smirnov, Martin Mladenov, Michael Gimelfarb, Scott Sanner",
      author_email="iliathesmirnov@gmail.com, mmladenov@google.com, mike.gimelfarb@mail.utoronto.ca, ssanner@mie.utoronto.ca",
      description="pyRDDLGym-impsmp: Implementation of the Importance Sampling for Policy Gradient algorithm that works with pyRDDLGym.",
      license="MIT License",
      packages=find_packages(),
      install_requires=[
          'pyRDDLGym>=2.0',
          'pyRDDLGym-jax'
        ],
      python_requires=">=3.8",
      package_data={'': ['*.cfg']},
      include_package_data=True,
      long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
      ],
      url="https://github.com/pyrddlgym-project/pyRDDLGym-impsmp",
)
