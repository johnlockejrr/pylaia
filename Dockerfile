FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Add python3 in cuda image
ENV DEBIAN_FRONTEND=non-interactive
RUN apt-get update -q -y && apt-get install -q -y --no-install-recommends python3-pip git

# Setup pylaia library
WORKDIR /src
COPY LICENSE pyproject.toml MANIFEST.in README.md /src/

COPY laia /src/laia

RUN pip install . --no-cache-dir
