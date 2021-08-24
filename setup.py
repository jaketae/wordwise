from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="wordwise",
    version="0.0.4",
    author="Jake Tae",
    author_email="jaesungtae@gmail.com",
    description="Keyword extraction using transformer-based language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jaketae/keyword",
    packages=find_packages(exclude=["docs", "tests"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
)
