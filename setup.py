from setuptools import find_packages, setup


INSTALL_REQUIRES = [
    "loguru==0.5.3",
    "psutil==5.8.0",
    "pydantic-1.8.2",
    "pyyaml==5.4.1",
    "tensorboard==2.5.0",
    "torch>=1.6.0",
    "tqdm==4.60.0",
]


with open("README.md") as f:
    long_description = f.read()


if __name__ == "__main__":
    setup(
        name="tez",
        description="tez - a simple pytorch trainer",
        entry_points={"console_scripts": ["tez=tez.cli.tez:main"]},
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Abhishek Thakur",
        author_email="abhishek4@gmail.com",
        url="https://github.com/abhishekkrthakur/tez",
        license="Apache License",
        package_dir={"": "src"},
        packages=find_packages("src"),
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        platforms=["linux", "unix"],
        python_requires=">3.5.2",
    )
