FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-1:m51

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install google-cloud-aiplatform && \
    pip install google-cloud-storage==1.32 && \
    pip install grpcio==1.38.0