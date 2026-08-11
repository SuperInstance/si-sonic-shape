from setuptools import setup, find_packages

setup(
    name="superinstance-sonic-shape",
    version="0.1.0",
    description="Confidence-to-music engine — maps agent confidence and emotion to musical parameters",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Lucineer / Casey DiGenaro",
    author_email="casey@superinstance.com",
    url="https://github.com/superinstance/sonic-shape",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[],  # pure Python
    extras_require={
        "live": ["aiohttp>=3.9", "websockets>=12.0"],
        "dev": ["pytest>=7.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
)
