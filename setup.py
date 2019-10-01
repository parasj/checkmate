from setuptools import setup

setup(
    name='remat',
    version='0.1.0',
    description='Optimal tensor rematerialization research project',
    packages=['remat'],  # find_packages()
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "pandas",
        "redis",
        "matplotlib",
        "seaborn",
        "tqdm",
        "ray>=0.7.5",
        "graphviz",
        "python-dotenv",
        "tensorflow>=2.0.0",
        "pytest",
        "keras_segmentation @ https://github.com/ajayjain/image-segmentation-keras/archive/master.zip#egg=keras_segmentation-0.2.0remat"
    ]
)
