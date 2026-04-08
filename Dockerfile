FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV SAM2_BUILD_CUDA=0
ENV LABEL_ANYTHING_SAM2_DIR=/opt/sam2
ENV LABEL_ANYTHING_MODEL_TYPE=base_plus

ARG SAM2_REF=main

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    libgl1 \
    libglib2.0-0 \
    python3-tk \
    tk \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY .docker-certs /tmp/extra-ca

RUN set -eux; \
    found=0; \
    if [ -d /tmp/extra-ca ]; then \
        for cert in /tmp/extra-ca/*; do \
            [ -f "$cert" ] || continue; \
            case "$cert" in \
                *.crt|*.pem) \
                    cp "$cert" "/usr/local/share/ca-certificates/$(basename "${cert%.*}").crt"; \
                    found=1; \
                    ;; \
            esac; \
        done; \
    fi; \
    if [ "$found" -eq 1 ]; then \
        update-ca-certificates; \
    fi

RUN pip install --upgrade pip setuptools wheel
RUN pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

RUN git clone https://github.com/facebookresearch/sam2.git /opt/sam2 \
    && cd /opt/sam2 \
    && git checkout "${SAM2_REF}" \
    && pip install -e .

RUN mkdir -p /opt/sam2/checkpoints \
    && wget -q -O /opt/sam2/checkpoints/sam2.1_hiera_tiny.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt \
    && wget -q -O /opt/sam2/checkpoints/sam2.1_hiera_small.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt \
    && wget -q -O /opt/sam2/checkpoints/sam2.1_hiera_base_plus.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt \
    && wget -q -O /opt/sam2/checkpoints/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install -e .

ARG BUILD_STAMP=dev
LABEL org.label_anything.build="${BUILD_STAMP}"

ENTRYPOINT ["label-anything"]
