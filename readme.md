# rl_waypoint_mrta
![Python](https://img.shields.io/badge/python-3.9-blue.svg)

This project uses DRL to address waypoint following task allocation and planning for multi-robot systems.

## 1. Dependencies
This software relies on Python 3.9. 

Please see the `requirements.txt` file for the details of dependencies. To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```

Otherwise, you can use the `Dockerfile` provided to build the environment. To build the docker image, run the following command:

```bash
docker build -t rlwaypointmrta:latest .
```

To run the docker image, run the following command:

```bash
docker run --rm -it  rlwaypointmrta:latest
```

The Pre-trained RL models for solving TSP problems are from: https://github.com/wouterkool/attention-learn-to-route

## 2. Usage

### 2.1. Training

```bash
python main.py --train
```

### 2.2. Evaluation

```bash
python main.py --eval --eval_dir trained_sessions/moe_mlp/rand_100-3/trained_model/batch31200.pt
```

### 2.3. Arguments
Arguments for training and evaluation that can be specified are defined in `arg_parser.py`. The docmentation of each argument is avaible by running the following command:

```bash
python main.py --help
```
### 2.4. Pre-trained models
Pre-trained models are available in the `trained_sessions` folder. The `trained_model` folder contains the trained model. Some relavent information about the training session is accessible using tesorboard. To run tensorboard, run the following command:

```bash
tensorboard --logdir=trained_sessions/$MODEL/$SESSION
```
