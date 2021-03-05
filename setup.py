import setuptools

setuptools.setup(
    name="src",
    version="0.0.1",
    description="Utility and helper functions",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19",
        "pandas>=1.1.5",
        "spacy==3.0.1",
        "spacy-alignments==0.7.2",
        "spacy-legacy==3.0.1",
    ],
)
