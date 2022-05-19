from setuptools import setup, find_packages

requirements = (
  'numpy>=1.19.1',
  'tensorflow>=2.2.0',
  'gpflow>=2.0.5',
  'tensorflow-probability>=0.9.0',
)

extra_requirements = {
  'Experiments': (
      'networkx',
      'matplotlib',
      'scipy',
      'tqdm',
      'osmnx',
      'contextily',
      'shapely',
      'pandas',
      'warn',
      'tables',
  ),
}

setup(name='graph_gaussian_process',
      version='0.1',
      license='MIT',
      packages=find_packages(exclude=["Experiments*"]),
      python_requires='>=3.6',
      install_requires=requirements,
      extras_require=extra_requirements)
