# setup.py
from setuptools import setup, find_packages

setup(
    name="neural-dsl",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "click",
        "lark-parser",
        "dash",
        "plotly",
        "flask",
        "flask-socketio",
        "requests",
        "torch",
        "tensorflow",
        "onnx",
        "psutil",
        "graphviz"
    ],
    entry_points={
        "console_scripts": ["neural=neural.cli:cli"]
    },
    author="Lemniscate-SHA-256/SENOUVO Jacques-Charles Gad",
    author_email="Lemniscate_zero@proton.me",
    description="A domain-specific language and debugger for neural networks",
    long_description=open("README.md").read() + "\n\n**Note**: This is an early release (v0.1.0) with known bugs—see GitHub for updates!",
    long_description_content_type="text/markdown",
    url="https://github.com/Lemniscate-SHA-256/Neural",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)