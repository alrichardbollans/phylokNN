from setuptools import setup, find_packages

setup(
    name='phylokNN',
    description='A package for phylogenetic nearest neighbour analysis',
    license='Attribution-NonCommercial-ShareAlike 4.0 International',
    packages=find_packages(include=['phylokNN', 'phyloAutoEncoder']),

    install_requires=[
        "pandas",
        "numpy",
        'scikit-learn'
    ],
    # *strongly* suggested for sharing
    version='1.0',
    long_description=open('README.md').read(),
)
