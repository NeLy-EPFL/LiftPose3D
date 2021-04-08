import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="liftpose",
    version="0.22",
    author="Adam Gosztolai & Semih Gunel",
    packages=["liftpose"],
    description="Monocular 3D pose Estimation on Laboratory Animals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeLy-EPFL/LiftPose3D"
)
