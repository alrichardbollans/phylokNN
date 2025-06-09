from setuptools import setup, find_packages

setup(
    name='phylokNN',
    description='A package for phylogenetic nearest neighbour analysis',
    author='Adam Richard-Bollans',
    url='https://github.com/alrichardbollans/phyloKNN',
    license='Attribution-NonCommercial-ShareAlike 4.0 International',
    packages=find_packages(include=['phylokNN']),

    install_requires=[
        "pandas",
        "numpy",
        'scikit-learn'
    ],
    # *strongly* suggested for sharing
    version='1.0',
    long_description=open('README.md').read(),
)
