# setup.py
from setuptools import setup, find_packages

setup(
    name="neural-dsl",
    version="0.2.6",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "click>=8.1.3",
        "flask>=3.0",
        "flask-cors>=3.1",
        "flask-httpauth>=4.4",
        "graphviz>=0.20",
        "lark>=1.1.5",
        "matplotlib<3.10",
        "networkx>=2.8.8",
        "numpy>=1.23.0",
        "psutil>=5.9.0",
        "pytest>=7.0.0",
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0",
        "pysnooper",
        "radon>=5.0"
    ],
    extras_require={
        "full": [
            "dash>=2.18.2",
            "dash-bootstrap-components>=1.0.0",
            "flask-socketio>=5.0.0",
            "plotly>=5.18",
            "torch>=1.10.0",
            "pygithub>=1.59",
            "selenium>=4.0",
            "optuna>=3.0",
            "fastapi>=0.68",
            "webdriver-manager",
            "tensorflow>=2.6",
            "huggingface_hub>=0.16",
            "transformers>=4.30",
            "torchvision>=0.15",
            "multiprocess>=0.70",
            "tweepy==4.15.0",
            "pandas>=1.3",
            "scikit-learn>=1.0",
            "scipy>=1.7",
            "seaborn>=0.11",
            "statsmodels>=0.13",
            "sympy>=1.9",
            "onnx>=1.10",
            "onnxruntime>=1.10"
        ]
    },
    entry_points={
        "console_scripts": ["neural=neural.__main__:cli"]
    },
    author="Lemniscate-SHA-256/SENOUVO Jacques-Charles Gad",
    author_email="Lemniscate_zero@proton.me",
    description="A domain-specific language and debugger for neural networks",
    long_description=open("README.md").read() + "\n\n**Note**: See v0.2.6 release notes for latest fixes and improvements!",
    long_description_content_type="text/markdown",
    url="https://github.com/Lemniscate-SHA-256/Neural",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
