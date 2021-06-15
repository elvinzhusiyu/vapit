# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash -e
project_name=img-seg-3d
component_name=vertex_base
image_name=gcr.io/${project_name}/${component_name} 
image_tag=v1
full_image_name=${image_name}:${image_tag}

# for me, cloud build doesn't work cross projects due to authentication issue 
# gcloud builds submit --tag ${full_image_name} .

docker build -t "${full_image_name}" .
docker push "$full_image_name"

# # Output the strict image name (which contains the sha256 image digest)
docker inspect --format="{{index .RepoDigests 0}}" "${IMAGE_NAME}"