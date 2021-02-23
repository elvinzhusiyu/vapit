#!/bin/bash -e
project_name=img-seg-3d
component_name=trainer
image_name=gcr.io/${project_name}/${component_name} 
image_tag=v1
full_image_name=${image_name}:${image_tag}

# for me, cloud build doesn't work cross projects due to authentication issue 
# gcloud builds submit --tag ${full_image_name} .

docker build -t "${full_image_name}" .
docker push "$full_image_name"

# # Output the strict image name (which contains the sha256 image digest)
docker inspect --format="{{index .RepoDigests 0}}" "${IMAGE_NAME}"