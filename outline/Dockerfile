FROM xwieme/ide:0.1

LABEL version="1.0" maintainer="Xander Wieme <xander.wieme@ugent.be>"

USER root 

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
        wget \
        texlive-full \
        biber \
        latexmk &&\
    apt-get clean &&\
    apt-get auto-remove -y

# USER user
