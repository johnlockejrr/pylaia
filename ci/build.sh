#!/bin/sh -e
# Build the tasks Docker image.
# Requires CI_PROJECT_DIR and CI_REGISTRY_IMAGE to be set.
# VERSION defaults to latest.
# Will automatically login to a registry if CI_REGISTRY, CI_REGISTRY_USER and CI_REGISTRY_PASSWORD are set.
# Will only push an image if $CI_REGISTRY is set.

if [ -z "$VERSION" ]; then
	VERSION=${CI_COMMIT_TAG:-latest}
fi

if [ -z "$VERSION" -o -z "$CI_PROJECT_DIR" -o -z "$CI_REGISTRY_IMAGE" ]; then
	echo Missing environment variables
	exit 1
fi

IMAGE_TAG="$CI_REGISTRY_IMAGE:$VERSION"

cd $CI_PROJECT_DIR
docker build -f Dockerfile . -t "$IMAGE_TAG"

# Publish the image on the main branch or on a tag
if [ "$CI_COMMIT_REF_NAME" = "$CI_DEFAULT_BRANCH" -o -n "$CI_COMMIT_TAG" ]; then
  if [ -n "$CI_REGISTRY" -a -n "$CI_REGISTRY_USER" -a -n "$CI_REGISTRY_PASSWORD" ]; then
    echo $CI_REGISTRY_PASSWORD | docker login -u $CI_REGISTRY_USER --password-stdin $CI_REGISTRY
    docker push $IMAGE_TAG
  else
    echo "Missing environment variables to log in to the container registry…"
  fi
else
  echo "The build was not published to the repository registry (only for default branch or tags)…"
fi
