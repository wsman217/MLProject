from setuptools import find_packages, setup

setup(
    name="canvas",
    packages=find_packages(include=["canvas"]),
    version="0.0.1",
    description="A library that provides a ui to draw in.",
    author="Weston",
    install_requires=["customtkinter", "os", "numpy", "tkinter", "Pillow"]
)
