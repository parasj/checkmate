from setuptools import setup

setup(
    name="checkmate",
    version="0.1.0",
    description="Checkmate prevents you from OOMing when training big deep neural nets",
    packages=["checkmate"],  # find_packages()
    python_requires=">=3.5",
    install_requires=[
        # "keras_segmentation @ https://github.com/ajayjain/image-segmentation-keras/archive/master.zip#egg=keras_segmentation-0.2.0remat",
        "cylp",
        "cvxpy",
        "numpy",
        "pandas",
        "toposort",
        "psutil",
    ],
    extras_require={"test": ["pytest", "tensorflow>=2.0.0", "matplotlib", "graphviz"]},
)
