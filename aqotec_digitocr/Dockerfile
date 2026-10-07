FROM python:3.11-slim-bookworm

ARG BUILD_VERSION
ARG BUILD_ARCH

LABEL \
  io.hass.version="${BUILD_VERSION}" \
  io.hass.type="app" \
  io.hass.arch="${BUILD_ARCH}"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pinned ONNX Model Zoo MNIST-12 classifier (MIT), used only as a digit classifier.
RUN curl -L --fail --retry 3 \
      -o /app/mnist-12.onnx \
      https://media.githubusercontent.com/media/onnx/models/4c46cd00fbdb7cd30b6c1c17ab54f2e1f4f7b177/validated/vision/classification/mnist/model/mnist-12.onnx \
    && echo "5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd  /app/mnist-12.onnx" | sha256sum -c -

COPY app.py /app/app.py

CMD ["python", "-u", "/app/app.py"]
