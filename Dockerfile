# This Dockerfile is currently hosted at https://hub.docker.com/r/dsteinmoeller/spins

FROM ubuntu:22.04

RUN apt-get update && apt-get -y install libfftw3-dev libsuitesparse-dev libopenblas-serial-dev clang-format \
        libboost-program-options-dev libopenmpi-dev curl zip unzip cmake python3-dev xxd build-essential g++ git

RUN curl -fsL https://github.com/blitzpp/blitz/archive/refs/tags/1.0.2.zip -o /blitz-1.0.2.zip && mkdir -p /work/blitz

WORKDIR /work/blitz
RUN unzip /blitz-1.0.2.zip && \
    cd blitz-1.0.2 && \
    rm /blitz-1.0.2.zip && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr .. && \
    make lib && \
    make install 


RUN mkdir -p /work/arma
WORKDIR /work/arma
RUN git clone  https://gitlab.com/conradsnicta/armadillo-code.git /work/arma && \
    git checkout 14.0.x && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr .. && \
    make -j8 && \
    make install