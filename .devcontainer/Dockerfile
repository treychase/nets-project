FROM mcr.microsoft.com/devcontainers/python:3.11

# PyMC/PyTensor compile the sampler with a C compiler at runtime, not just at
# install time, so this needs to be present in the container, not just on the
# machine that built the image.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt
