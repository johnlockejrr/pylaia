FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /src

# Install PyLaia as a package
COPY laia laia
COPY requirements.txt doc-requirements.txt setup.py laia/VERSION README.md ./

RUN pip install . --no-cache-dir
