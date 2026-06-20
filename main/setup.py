from setuptools import setup, find_packages

setup(
    name="trackid",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "librosa",
    ],
    entry_points={
        'console_scripts': [
            'trackid=trackid.cli:main',
        ],
    },
)