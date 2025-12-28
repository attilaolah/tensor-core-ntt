# syntax=docker/dockerfile:1

ARG CUDA_VERSION=13.1.0
ARG OS_VERSION=ubuntu24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-${OS_VERSION}

RUN rm -f /etc/apt/apt.conf.d/docker-clean \
  ; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --upgrade --yes \
    ca-certificates \
    clang-format \
    cmake \
    cmake-format \
    git \
    less \
    libboost-dev \
    libgtest-dev \
    nano \
    ninja-build \
    openssh-client \
  && useradd --create-home --shell /bin/bash user
