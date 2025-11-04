FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive
ARG TEST_ENV

WORKDIR /app

RUN conda update conda -y

RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    apt-get -y update \
    && apt-get install -y git \
    && apt-get install -y wget \
    && apt-get install -y curl \
    && apt-get install -y g++ freeglut3-dev build-essential libx11-dev \
    libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev ninja-build \
    && apt-get -y install ffmpeg libsm6 libxext6 libffi-dev python3-dev python3-pip gcc

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/.cache \
    PORT=9090 \
    WORKERS=2 \
    THREADS=4 \
    CUDA_HOME=/usr/local/cuda \
    GROUNDINGDINO_REPO_PATH=/GroundingDINO

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX;8.9;9.0"

# install base requirements
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

# install model requirements
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip3 install -r requirements.txt

# install test requirements if needed
COPY requirements-test.txt .
# build only when TEST_ENV="true"
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
      pip3 install -r requirements-test.txt; \
    fi

WORKDIR /app

COPY . ./

# Install Grounding DINO from source and download checkpoints
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ ! -d "${GROUNDINGDINO_REPO_PATH}" ]; then \
        git clone https://github.com/IDEA-Research/GroundingDINO.git ${GROUNDINGDINO_REPO_PATH}; \
    fi \
    && pip3 install -e ${GROUNDINGDINO_REPO_PATH}

RUN mkdir -p ${GROUNDINGDINO_REPO_PATH}/weights \
    && cd ${GROUNDINGDINO_REPO_PATH}/weights \
    && wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
    && wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

# Copy any project-provided checkpoints into the Grounding DINO weights directory
COPY gdino_swint_darpa-ir-v1-1k_20_1.pth ${GROUNDINGDINO_REPO_PATH}/weights/

ENV PYTHONPATH=/app

CMD ["/app/start.sh"]
