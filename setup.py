from setuptools import setup, find_packages

setup(
name='playervectors',
version='0.1.0',
author='Raphael Senn',
author_email='raphaelsenn@outlook.com',
description='A package to summarise the playing styles of individual football players.',
package_dir={'': 'src'},
packages=find_packages('src'),
install_requires=[
    'seaborn',
    'numpy',
    'scikit-learn',
    'pandas',
    'matplotlib',
    'scipy',
],
extras_require={
    # Pitch drawing in playervectors.visualize. Optional, because databallpy
    # pins numpy>=2, which conflicts with e.g. numba and socceraction.
    'plot': ['databallpy'],
},
classifiers=[
'Programming Language :: Python :: 3',
'License :: OSI Approved :: MIT License',
'Operating System :: OS Independent',
],
python_requires='>=3.12',
)