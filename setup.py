from setuptools import find_packages, setup

version = {}

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

with open("wordwise/__version__.py", "r") as version_file:
    exec(version_file.read(), version)

with open("requirements.txt", "r") as requirements_file:
    install_requires = requirements_file.read().splitlines()

setup(
    name="wordwise",
    version=version["version"],
    author="Jake Tae",
    author_email="jaesungtae@gmail.com",
    description="Keyword extraction using transformer-based language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jaketae/wordwise",
    packages=find_packages(exclude=["docs", "tests"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=install_requires,
)
