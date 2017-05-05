"""A setuptools based setup module.

See:
https://python-packaging.readthedocs.io/en/latest/everything.html
"""

from setuptools import setup


def readme():
    """Loads the content of the readme file."""
    with open('README.md') as rmfile:
        return rmfile.read()


setup(name='tensorlight',
      version='1.0.0',
      description='TensorLight - A high-level framework for TensorFlow.',
      long_description=readme(),
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      keywords='tenforflow library framework dataset deep-learning neural-network machine-learning',
      url='https://github.com/bsautermeister/tensorlight',
      author='Benjamin Sautermeister',
      author_email='mail@bsautermeister.de',
      license='MIT',
      packages=['tensorlight'],
      install_requires=[
          'tensorflow>=1.0.0'
          'numpy',
          'matplotlib',
          'opencv-python',
          'scipy',
          'scikit-image',
          'sk-video',
          'moviepy',
          'rarfile',
          'h5py',
          'jsonpickle'
      ],
      include_package_data=True,
      zip_safe=False)
