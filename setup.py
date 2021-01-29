from setuptools import setup, find_packages

# TODO - install_requires, python_requires='>=3.6', extras_require

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