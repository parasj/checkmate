from setuptools import setup

setup(
    name='remat',
    version='0.1.0',
    description='Optimal tensor rematerialization research project',
    packages=['remat'],  # find_packages()
    python_requires='>=3.6',
    install_requires=[
        "matplotlib",  # this is only used once in the core remat package
        "numpy",
        "pandas",
        "pytest",
        "tensorflow>=2.0.0",
        "toposort",
    ],
    extras_require={
        'gpu': ['tensorflow-gpu>=2.0.0'],
        'eval': [
            "keras_segmentation @ https://github.com/ajayjain/image-segmentation-keras/archive/master.zip#egg=keras_segmentation-0.2.0remat",
            "graphviz",
            "python-dotenv",
            "ray>=0.7.5",
            "redis",
            "scipy",
            "seaborn",
            "tqdm",
        ],
        'test': [
            "keras_segmentation @ https://github.com/ajayjain/image-segmentation-keras/archive/master.zip#egg=keras_segmentation-0.2.0remat",
            "graphviz",
        ]
    }
)
