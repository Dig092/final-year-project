# Define a build argument for the target platform (CPU or GPU)
ARG TARGET=cpu

# Use conditional base image selection based on the TARGET argument
FROM python:3.11.9-slim AS cpu-base
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel AS gpu-base

# Use a multi-stage build to select the correct base image
FROM ${TARGET}-base as final

ENV DEBIAN_FRONTEND=noninteractive

ENV KAGGLE_USERNAME=vikasramachandra
ENV KAGGLE_KEY=f24a0e3c39e05518e41f50f12757ae73

WORKDIR /agent/

RUN apt update -y&& \
    apt install gcc g++ python3-opencv ssh git libgl1-mesa-dev -y

COPY requirements.txt /agent/

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install -r requirements.txt

COPY src/app_claude.py /agent/

COPY src/auth.py /agent/

EXPOSE 8000

ENTRYPOINT [ "python3","app_claude.py"]
