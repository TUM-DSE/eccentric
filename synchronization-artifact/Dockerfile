FROM ubuntu:22.04

WORKDIR /qec
ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt ./

# Python packages
RUN apt-get update && apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y python3.9 python3-distutils python3-pip python3-apt

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

# Other SW
RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get install -y git vim jq bc make automake cmake libnuma-dev \
    && apt-get install -y rsync htop curl build-essential \
    && apt-get install -y pkg-config libffi-dev libgmp-dev \
    && apt-get install -y libssl-dev libtinfo-dev libsystemd-dev \
    && apt-get install -y zlib1g-dev make g++ wget libncursesw5 libtool autoconf \
    && apt-get clean