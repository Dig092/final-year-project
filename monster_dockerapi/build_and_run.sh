set -e

IMAGE_NAME="qblockrepo/neo_docker_mgmt_api:latest"

docker build -t $IMAGE_NAME .

docker run -it -p 8000:8000 -v /var/run/docker.sock:/var/run/docker.sock $IMAGE_NAME  

docker push $IMAGE_NAME