# run this script from the root directory of the project

# run this command after updating Dockerfile
docker login
# build for single platform
docker build -f image.dockerfile -t ivanjaenm/tf4ot:v1 . --push
# built for both platforms
docker buildx build --platform linux/amd64,linux/arm64 -t ivanjaenm/decoder4ot:v0 -f scripts/image.dockerfile . --push
#docker system prune --all --force --volumes

# run using docker
docker run --platform linux/arm64 -it -v .:/home/decoder4ot ivanjaenm/decoder4ot:v0 bash
#python3 -m optformer.decoding_regression.models_test