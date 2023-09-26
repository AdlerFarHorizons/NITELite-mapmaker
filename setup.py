'''Installation script for NITELite mapmaker.
'''

import setuptools

setuptools.setup(
    name="nitelite_mapmaker",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'pytest',
        'jupyterlab',
        'jupyter_contrib_nbextensions',
    ],
)
