#
docker compose -f ./langfuse/docker-compose.yml up -d

docker compose -f ./ollama/docker-compose.yml up -d

docker compose -f ./store/docker-compose.yml up -d
