# cacic2019-handshapes
Experiments for the article "Handshape Recognition for Small Dataset" 

## Content

- [Quickstart](#quickstart)
- [Datasets](#datasets)
- [Models](#models)
  - [Prototypical Networks for Few-shot Learning](#prototypical-networks-for-few-shot-learning)
    - [Training]
    - [Evaluating]
  - [Dense Net](#dense-net)
    - [Training]
    - [Evaluating]
- [Results](#results)



## Quickstart

To start the docker container execute the following command

```sh
$ ./bin/start [-n <string>] [-t <tag-name>] [--sudo] [--build]
```

```
<tag-name> = cpu | devel-cpu | gpu
```

For example:

```sh
$ ./bin/start -n myContainer -t gpu --sudo --build
```

Once the docker container is running it will execute the contents of the /bin/execute file.

You can execute

```sh
$ docker exec -it <container-id> /bin/sh -c "[ -e /bin/bash ] && /bin/bash || /bin/sh"
```
to access the running container's shell.

## Datasets

In our paper we used the datasets RWTH-Phoenix, LSA16 and CIARP. We used the library (https://github.com/midusi/handshape_datasets) to fetch the datasets.

## Models

### Prototypical Networks for Few-shot Learning

Tensorflow v2 implementation of NIPS 2017 Paper _Prototypical Networks for Few-shot Learning_.

Implementation based on [protonet](https://github.com/ulises-jeremias/prototypical-networks-tf).

#### Training

Run the following command to run training on `<config>` with default parameters.

```sh
$ ./bin/protonet --mode train --config <config>
```

`<config> = lsa16 | rwth | ciarp`

#### Evaluating

To run evaluation on a specific dataset

```sh
$ ./bin/protonet --mode eval --config <config>
```

`<config> = lsa16 | rwth | ciarp`


### Dense Net

We implemented Densenet using squeeze and excitation layers in tensorflow 2 for our experiments. To see its implementation go to (https://github.com/okason97/DenseNet-Tensorflow2).

For more information about densenet please refer to the original paper (https://arxiv.org/abs/1608.06993).

#### Training

To train Densenet on all the datasets and search for the best configuration execute `/bin/densenet_train_all.py`. This will give you the results of each configuration in the folder `/results` and the summary of the training of each configuration on `/results/summary.csv`.
If you want to train densenet with your own configurations you can use `/src/dense_net/train.py`. You can customize your training modifying the parameters of train.py. Use it in your python code in the following way

```python
from src.dense_net.train import train_densenet
train_densenet(dataset_name = "rwth", rotation_range = 10, width_shift_range = 0.10,
               height_shift_range = 0.10, horizontal_flip = True, growth_rate = 128,
               nb_layers = [6,12], reduction = 0.0, lr = 0.001, epochs = 400,
               max_patience = 25, batch_size= 16, checkpoints = False, weight_classes = False,
               train_size=None, test_size=None)
```

To use your own datasets you can add them to `/src/dense_net/datasets/loader.py` and call `train.py` using the name you chose.

#### Evaluating

For evaluation you can use `/src/dense_net/eval.py`

```python
from src.dense_net.eval import eval_densenet
eval_densenet(dataset_name = "rwth", growth_rate = 128, nb_layers = [6,12],
              reduction = 0.0, batch_size = 16, weight_classes = False, model_path = "")
```

## Results

In the `/results` directory you can find the results of a training processes using a `<model>` on a specific `<dataset>`:

```
.
├─ . . .
├─ results
│  ├─ <dataset>                            # results for an specific dataset.
│  │  ├─ <model>                           # results training a <model> on a <dataset>.
│  │  │  ├─ models                         # ".h5" files for trained models.
│  │  │  ├─ results                        # ".csv" files with the different metrics for each training period.
│  │  │  ├─ summaries                      # tensorboard summaries.
│  │  │  ├─ config                         # optional configuration files.
│  │  └─ └─ <dataset>_<model>_results.csv  # ".csv" file in which the relationships between configurations, models, results and 
summaries are listed by date.
│  └─ summary.csv                          # contains the summary of all the training
└─ . . .
```

where

```
<dataset> = lsa16 | rwth | . . .
<model> = dense-net | proto-net
```

To run TensorBoard, use the following command:

```sh
$ tensorboard --logdir=./results/<dataset>/<model>/summaries
```
