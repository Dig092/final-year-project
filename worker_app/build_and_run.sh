set -e

docker build -t neo_docker_worker_cpu .

docker run -dt -p 8000:8000 neo_docker_worker_cpu