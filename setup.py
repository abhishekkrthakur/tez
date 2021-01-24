from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()


if __name__ == "__main__":
    setup(
        name="tez",
        version="0.1.1",
        description="tez - a simple pytorch trainer",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Abhishek Thakur",
        author_email="abhishek4@gmail.com",
        url="https://github.com/abhishekkrthakur/tez",
        license="Apache License",
        packages=find_packages(),
        include_package_data=True,
        install_requires=["torch>=1.6.0"],
        platforms=["linux", "unix"],
        python_requires=">3.5.2",
    )
