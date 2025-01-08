from setuptools import setup, find_packages

setup(
    name='phylonn',

    packages=find_packages(),

    install_requires=[
        "pandas",
        "numpy",
        'scikit-learn'
    ],
    # *strongly* suggested for sharing
    version='1.0',
    long_description=open('README.md').read(),
)
