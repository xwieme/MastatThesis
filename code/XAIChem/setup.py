import setuptools

setuptools.setup(
    name="XAIChem",
    version="0.0.1",
    author="xwieme",
    description="Machine learning package with explainable techniques for chemistry",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    package_dir={"": "."},
    packages=setuptools.find_namespace_packages(where="."),
    python_requires=">= 3.11",
)