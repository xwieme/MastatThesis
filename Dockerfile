# FROM nvidia/cuda:11.8.0-base-ubuntu22.04
FROM xwieme/ide:0.1

LABEL version="1.0" maintainer="Xander Wieme <xander.wieme@ugent.be>"

USER root 

ARG CONDA_DIR=/opt/conda/
ENV PATH=$CONDA_DIR/bin:$CONDA_DIR/envs/lab/bin:$PATH
# Environment variable used by apptainer to load the default conda env
ENV BASH_ENV=/opt/etc/bashrc

RUN apt-get update &&\
    apt-get install -y \
        wget \
        git &&\
    apt-get clean

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh &&\
    bash /tmp/miniconda.sh -b -u -p $CONDA_DIR &&\
    rm /tmp/miniconda.sh


# Create conda environment and setup bashrc to
# load the conda environment lab as default
RUN conda create -n lab python=3.11 -y &&\
    conda install -y -n lab \
        pytorch=2.1.1 \
        pytorch-cuda=11.8 \
        pyg=2.4.0 \
        wandb=0.15.12 \
        jupyterlab=4.0.8 \
        ipython=8.16.1 \
        pandas=2.1.1 \
        rdkit=2023.09.2 \
        plotly=5.18.0 \
        pyright \
        black \
        -c pytorch \
        -c pyg \
        -c nvidia \
        -c plotly \
        -c anaconda \
        -c conda-forge &&\
    mkdir -p /opt/etc &&\
    echo "#!/bin/bash\n\n# Script to activate conda environment" > ~/.bashrc &&\
    conda init bash &&\
    echo "\nconda activate lab" >> ~/.bashrc &&\
    cp ~/.bashrc /opt/etc/bashrc &&\
    conda clean -tipfay &&\
    chown --recursive user:user $CONDA_DIR 

USER user
