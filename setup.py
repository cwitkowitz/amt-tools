from setuptools import setup, find_packages

# TODO - default paths (generated) may be no bueno for usage with pip installation
# TODO - sacred in extras require specifying dependencies for /examples?

setup(
    name='amt-tools',
    url='https://github.com/cwitkowitz/amt-tools',
    author='Frank Cwitkowitz',
    author_email='fcwitkow@ur.rochester.edu',
    # TODO - why does 'pip install -e' consider generated/examples as packages?
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=['numpy', 'librosa', 'torch', 'matplotlib', 'sacred', 'mir_eval',
                      'jams', 'mido', 'requests', 'tqdm', 'tensorboard', 'tensorboardX',
                      'scipy', 'pandas', 'mirdata', 'sounddevice', 'pynput'],
    #scripts=['examples/of_1.py', 'examples/of_2.py', 'examples/tabcnn.py'],
    version='0.3.0',
    license='MIT',
    description='Machine learning tools and framework for automatic music transcription',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)
