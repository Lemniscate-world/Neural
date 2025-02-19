from setuptools import setup, find_packages

setup(
    name='neural',
    version='0.1.0',
    description='A neural network framework with native .neural and .nr file support',
    author='Lemniscate-SHA-256',
    packages=find_packages(where='.', exclude=["tests*", "docs"]),
    install_requires=[
        'lark',
        'click',
        # other dependencies like torch, tensorflow, etc. as needed
    ],
    entry_points={
        'console_scripts': [
            'neural=cli:cli',  # This creates the "neural" CLI command.
        ],
    },
)
