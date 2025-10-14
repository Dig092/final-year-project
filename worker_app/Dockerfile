# Define a build argument for the target platform (CPU or GPU)
ARG TARGET=gpu

# Use conditional base image selection based on the TARGET argument
FROM python:3.11.9-slim AS cpu-base
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel AS gpu-base

# Use a multi-stage build to select the correct base image
FROM ${TARGET}-base as final

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /agent/

RUN apt update -y&& \
    apt install curl gcc g++ python3-opencv ssh git libgl1-mesa-dev -y

COPY requirements.txt /agent/

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install -r requirements.txt

COPY src/app_claude.py /agent/

COPY src/auth.py /agent/

RUN apt install -y gcc g++ zip unzip cmake 

RUN echo '#!/bin/bash' > start.sh \
    && echo 'curl -fsSL https://code-server.dev/install.sh | sh' >> start.sh \
    && echo 'code-server --auth none --bind-addr 0.0.0.0:80' >> start.sh \
    && chmod +x start.sh

RUN echo '#!/bin/bash' > run.sh \
    && echo './start.sh &' >> run.sh \
    && echo 'python3 app_claude.py' >> run.sh \
    && chmod +x run.sh

EXPOSE 8000 80

ENTRYPOINT [ "./run.sh"]
