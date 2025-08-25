#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 [cpu|gpu]"
  exit 1
fi

# Set the target based on the argument
TARGET=$1

# Validate the target
if [ "$TARGET" != "cpu" ] && [ "$TARGET" != "gpu" ]; then
  echo "Invalid target. Please use 'cpu' or 'gpu'."
  exit 1
fi

# Set the image name
IMAGE_NAME="qblockrepo/neo_agent_worker:${TARGET}-latest"

# Build the Docker image
echo "Building Docker image for $TARGET..."
docker build --build-arg TARGET=$TARGET -t $IMAGE_NAME .

if [ $? -ne 0 ]; then
  echo "Failed to build Docker image."
  exit 1
fi

# Run the Docker container
#echo "Running Docker container for $TARGET..."
#docker run -it $IMAGE_NAME

if [ $? -ne 0 ]; then
  echo "Failed to run Docker container."
  exit 1
fi

echo "Docker container is running on port 8000."

#docker push $IMAGE_NAME
