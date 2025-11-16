FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

# Install Python 3.11 and system deps
RUN set -ex \
    && apt-get -yqq update \
    && DEBIAN_FRONTEND=noninteractive apt-get -yqq install \
        curl \
        rsync \
        git \
        nvtop \
        python3.11 \
        python3.11-venv \
        python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3.11 /tmp/get-pip.py \
    && rm /tmp/get-pip.py

# Make python3 / pip3 point to Python 3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/local/bin/pip /usr/bin/pip3

# From here on, ALWAYS use python3 -m pip (this is Python 3.11 now)
COPY ./requirements.txt /tmp/requirements.txt

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install your normal requirements
# (torch pinned in requirements.txt; DO NOT include flash-attn there)
RUN python3 -m pip install -r /tmp/requirements.txt

# Then install flash-attn for Python 3.11 / CUDA 12 / Torch 2.8
RUN python3 -m pip install --no-build-isolation --no-cache-dir \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl