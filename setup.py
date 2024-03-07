from setuptools import setup, find_packages

setup(
    name="bhscan",
    version="0.1.0",
    description="A Python-based application for processing and analyzing WPA census card images for the Bunker Hill Refrain project.",
    author="Elea Zhong",
    author_email="eleaz@usc.edu",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "opencv-python-headless",
        "pytesseract",
        "argparse",
        "numpy",
        "scikit-learn",
        "Pillow",
        "tqdm",
        "pandas",
        "matplotlib",
        "Wand",
        "pydantic",
    ],
    entry_points={
        'console_scripts': [
            'bhscan=src.main:main',
        ],
    },
)
