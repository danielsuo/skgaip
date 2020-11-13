from setuptools import setup
from setuptools import find_packages

setup(
    name="skgaip",
    packages=find_packages(),
    install_requires=[
        "timecast-nightly",
        "sklearn",
        "matplotlib",
        "tensorflow-gpu==2.3.1",
        "keras",
        "pathos",
        "pandas",
        "numpy",
        "scipy",
        "cython",
        "jax",
        "jaxlib",
        "flax",
        "binpacking",
        "pixiedust",
        "psycopg2",
        "sqlalchemy",
    ],
)
