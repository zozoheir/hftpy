from setuptools import setup, find_packages

setup(
    name='hftpy',
    version='0.1',
    description='HFTpy first release',
    packages=find_packages(),
    author='Othmane Zoheir',
    author_email='othmane@rumorz.io',
    url='',
    install_requires=[
        'numpy',
        'pandas'
    ]
)
