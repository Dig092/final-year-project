set -e

IMAGE_NAME="qblockrepo/neo_docker_mgmt_api:latest"

docker build -t $IMAGE_NAME . -f Dockerfile

docker run -dt -p 8000:8000 --network=my_network -v /var/run/docker.sock:/var/run/docker.sock -v /home/$USER/.aws/:/root/.aws/ $IMAGE_NAME

#docker run -it -p 8000:8000 -v /var/run/docker.sock:/var/run/docker.sock -v /home/ubuntu/.aws/:/root/.aws/ qblockrepo/neo_docker_mgmt_api:latest

docker push $IMAGE_NAME
