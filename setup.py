from setuptools import find_packages, setup


with open("README.md") as f:
    long_description = f.read()


INSTALL_REQUIRES = [
    "loguru>=0.6.0",
    "accelerate>=0.12.0",
    "tqdm>=4.64.0",
]


if __name__ == "__main__":
    setup(
        name="tez",
        description="tez - a simple pytorch trainer",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Abhishek Thakur",
        url="https://github.com/abhishekkrthakur/tez",
        license="Apache License",
        packages=find_packages(),
        include_package_data=True,
        platforms=["linux", "unix"],
        python_requires=">=3.7",
        install_requires=INSTALL_REQUIRES,
    )
