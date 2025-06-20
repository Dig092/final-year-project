# Define a build argument for the target platform (CPU or GPU)
ARG TARGET=cpu

# Use conditional base image selection based on the TARGET argument
FROM python:3.11.9-slim AS cpu-base
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel AS gpu-base

# Use a multi-stage build to select the correct base image
FROM ${TARGET}-base as final

WORKDIR /agent/

COPY requirements.txt /agent/

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install -r requirements.txt

COPY src/app.py /agent/

EXPOSE 8000

ENTRYPOINT [ "python3","app.py"]