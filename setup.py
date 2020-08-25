import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch-deploy", # Replace with your own username
    version="0.0.6",
    author="Owen Mo, Fiona Xie, Hulbert Zhang",
    author_email="mochangheng@gmail.com, fionax@andrew.cmu.edu, hzeng012@ucr.edu",
    description="Serving pytorch models on an API in one line.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mochangheng/pytorch-deploy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "uvicorn",
        "fastapi",
        "pydantic",
        "requests",
        "jinja2",
        "python-multipart"
    ],
    python_requires='>=3.6',
)