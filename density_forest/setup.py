from setuptools import setup
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    print(long_description)

setup(name='density_forest',
      version='0.4.1',
      description='Density Forest library for confidence estimation and novelty detection',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/CyrilWendl/SIE-Master',
      author='Cyril Wendl',
      author_email='cyrilwendl@gmail.com',
      license='MIT',
      install_requires=['numpy', 'matplotlib', 'scipy', 'tqdm', 'Cython', 'scikit-image', 'pandas', 'torchvision',
                        'pytorch', 'keras', 'tensorflow', 'keras', 'joblib', 'pip', 'sklearn'],
      packages=['density_forest', 'baselines', 'helpers', 'keras_helpers'],
      zip_safe=False,
      )
