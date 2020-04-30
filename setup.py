from setuptools import setup
from setuptools import find_packages

setup(
    name="skgaip",
    packages=find_packages(),
    install_requires=[
        "timecast-nightly",
        "sklearn",
        "matplotlib",
        "tensorflow-gpu==1.15",
        "keras",
        "pathos"
        "pandas",
        "numpy",
        "scipy",
        "cython",
        "jax",
        "jaxlib",
        "flax",
    ]

)
