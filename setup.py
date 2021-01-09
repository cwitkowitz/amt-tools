from setuptools import setup, find_packages

setup(
    name = 'amt_models',
    url = 'https://github.com/cwitkowitz/transcription-models',
    author = 'Frank Cwitkowitz',
    author_email = 'fcwitkow@ur.rochester.edu',
    packages = find_packages(),
    install_requires = ['numpy', 'librosa', 'torch'],
    version = '0.1.0',
    license = 'MIT',
    description = 'Music Transcription Tools',
    long_description = open('README.md').read()
)