from setuptools import setup, find_packages


PROJECT_REPO = 'https://gitlab.uni.lu/xcit/icom/pyvrc'

with open('requirements.txt') as f:
  REQUIRED_PACKAGES = f.read().splitlines()

with open("README.md", "r") as f:
  LONG_DESCRIPTION = f.read()

setup(name='pyvrc',
      version='2020.11.1',
      description='',
      long_description=LONG_DESCRIPTION,
      long_description_content_type="text/markdown",
      keywords='information-theory psychology variable-rate-coding',
      url=PROJECT_REPO,
      project_urls={
          "Bug Tracker": f"{PROJECT_REPO}/issues",
          "Documentation": "/docs",
          "Source Code": PROJECT_REPO,
          "Trello Board": ""
      },
      package_dir={'': 'python'},
      packages=find_packages(where='python'),
      python_requires='>=3.7',
      install_requires=REQUIRED_PACKAGES,
      )
