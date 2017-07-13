FROM ubuntu:latest

RUN apt-get update -yqq  && apt-get install -yqq \
    wget \
    bzip2 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Configure environment
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# install miniconda and python 3
RUN wget -O miniconda.sh \
  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && bash miniconda.sh -b -p /work/miniconda \
  && rm miniconda.sh

ENV PATH="/work/bin:/work/miniconda/bin:$PATH"

# install long-term requirements
RUN conda update -y python conda && \
    conda install -y \
    pip \
    setuptools \
    jupyter \
    numpy \
    scipy \
    pandas \
    bokeh \
    dask \
    distributed \
    scikit-learn \
    click

RUN pip install traces
RUN pip install git+git://github.com/thomasgreg/dask-searchcv.git@273defe276c0318989f0eb6c56ec9a9d5fa035c7

# copy source to image and python setup.py develop
COPY . /work/

# do python setup.py develop
RUN cd /work
