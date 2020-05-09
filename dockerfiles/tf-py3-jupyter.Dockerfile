ARG DOCKER_ENV=latest

FROM tensorflow/tensorflow:${DOCKER_ENV}
# DOCKER_ENV are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)

ARG DOCKER_ENV
ARG NODE_VERSION=12.x

ADD . /develop

# Needed for string testing
SHELL ["/bin/bash", "-c"]

RUN apt-get update -q && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    apt-get install -y git nano graphviz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install models, scripts, protonet scripts and handshape-recognition utils
# Clone and install protonet
# Clone and install handshape datasets
RUN pip install --upgrade pip && \
    pip3 install -e /develop && \
    pip3 install -U tensorflow && \
    pip3 install gdown==3.10.0 && \
    git clone https://github.com/midusi/handshape_datasets.git /tf/lib/handshape_datasets && \
    git -C /tf/lib/handshape_datasets checkout 0dc9fcd63cfdf128c00535da9d9fca31b903c4eb && \
    git clone https://github.com/ulises-jeremias/prototypical-networks-tf.git /tf/lib/prototypical-networks-tf && \
    git clone https://github.com/okason97/DenseNet-Tensorflow2 /tf/lib/DenseNet-Tensorflow2 && \
    pip3 install -e /tf/lib/prototypical-networks-tf && \
    pip3 install -e /tf/lib/DenseNet-Tensorflow2 && \
    pip3 install -e /tf/lib/handshape_datasets && \
    pip3 install -U tensorflow && \
    pip3 install tensorflow_datasets && \
    pip3 install seaborn eli5 shap pydot pdpbox sklearn opencv-python IPython prettytable py7zr

# Default dir for handshape datasets lib - use /data instead
RUN mkdir -p /.handshape_datasets && \
    chmod -R a+rwx /.handshape_datasets && \
    mkdir -p /.cache && \
    chmod -R a+rwx /.cache

## Install node, yarn and hand-cropper dependencies
### install nodejs and yarn packages from nodesource and yarn apt sources
# RUN echo "deb https://deb.nodesource.com/node_${NODE_VERSION} stretch main" > /etc/apt/sources.list.d/nodesource.list && \
#    echo "deb https://dl.yarnpkg.com/debian/ stable main" > /etc/apt/sources.list.d/yarn.list && \
#    curl -sS https://deb.nodesource.com/gpgkey/nodesource.gpg.key | apt-key add - && \
#    curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add - && \
#    apt-get update -qq && \
#    apt-get install -qq -y --no-install-recommends nodejs yarn && \
#    rm -rf /var/lib/apt/lists/* && \
#    git clone https://github.com/okason97/hand-cropper.git /tf/lib/hand-cropper && \
#    yarn --cwd /tf/lib/hand-cropper

WORKDIR /develop
