from setuptools import setup

# TODO - default paths (generated) may be no bueno for usage with pip installation
# TODO - sacred in extras require specifying dependencies for /examples?

setup(
    name='amt-tools',
    url='https://github.com/cwitkowitz/amt-tools',
    author='Frank Cwitkowitz',
    author_email='fcwitkow@ur.rochester.edu',
    packages=['amt_tools'],
    python_requires='>=3.7',
    install_requires=['numpy', 'librosa', 'torch', 'matplotlib',
                      'sacred', 'mir_eval', 'jams', 'mido', 'requests',
                      'tqdm', 'tensorboardX', 'scipy', 'pandas', 'mirdata'],
    #scripts=['examples/of_1.py', 'examples/of_2.py', 'examples/tabcnn.py'],
    version='0.1.3',
    license='MIT',
    description='Machine learning tools and framework for automatic music transcription',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)
