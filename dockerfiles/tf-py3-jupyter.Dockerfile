ARG DOCKER_ENV=cpu

FROM ulisesjeremias/tf-docker:${DOCKER_ENV}-jupyter
# DOCKER_ENV are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)

ARG DOCKER_ENV

ADD . /develop

# Needed for string testing
SHELL ["/bin/bash", "-c"]

RUN apt-get update -q && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    apt-get install -y git nano graphviz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install models, scripts, proto-net scripts and handshape-recognition utils
# Clone and install proto-net
# Clone and install handshape datasets
RUN pip install --upgrade pip && \
    pip3 install -e /develop && \
    pip3 install densenet && \
    git clone https://github.com/midusi/handshape_datasets.git /tf/lib/handshape_datasets && \
    pip3 install -e /tf/lib/handshape_datasets && \
    git clone --branch=develop https://github.com/ulises-jeremias/prototypical-networks-tf.git /tf/lib/prototypical-networks-tf && \
    pip3 install -e /tf/lib/prototypical-networks-tf && \
    pip3 install -U tensorflow && \
    pip3 install tensorflow_datasets && \
    pip3 install seaborn eli5 shap pydot pdpbox sklearn opencv-python IPython && \
    if [[ "$DOCKER_ENV" = "gpu" ]]; then echo -e "\e[1;31mINSTALLING GPU SUPPORT\e[0;33m"; pip3 install -U tf-nightly-gpu-2.0-preview tb-nightly; fi

# Default dir for handshape datasets lib - use /data instead
RUN mkdir -p /.handshape_datasets && \
    chmod -R a+rwx /.handshape_datasets

WORKDIR /develop
