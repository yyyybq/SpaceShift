ARG CUDA_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ARG NVIDIA_VERSION
ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get -y install \
    wget git ca-certificates bzip2 \
    libvulkan1 pciutils kmod vim \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Miniconda
# -----------------------------
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
 && rm /tmp/miniconda.sh

ENV PATH=${CONDA_DIR}/bin:$PATH

# -----------------------------
# App layout
# -----------------------------
ENV APP_HOME=/app
WORKDIR ${APP_HOME}

# -----------------------------
# Create conda env from environment.yml
# -----------------------------
COPY environment.yml /tmp/environment.yml
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda env create -f /tmp/environment.yml -n sceneshift_env \
 && conda clean -afy

# Make sceneshift_env the default python environment
ENV PATH=${CONDA_DIR}/envs/sceneshift_env/bin:$PATH

# -----------------------------
# AI2-THOR + NVIDIA setup
# -----------------------------
COPY scripts/install_nvidia.sh /app/install_nvidia.sh

RUN python -c "import os; import ai2thor.build; ai2thor.build.Build( \
    'CloudRendering', \
    ai2thor.build.DEFAULT_CLOUDRENDERING_COMMIT_ID, \
    False, \
    releases_dir=os.path.join(os.path.expanduser('~'), '.ai2thor/releases') \
).download()"

RUN NVIDIA_VERSION=$NVIDIA_VERSION /app/install_nvidia.sh

# -----------------------------
# Code will be mounted here
# -----------------------------
WORKDIR /workspace/spatial-scene-variations

COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["/bin/bash"]