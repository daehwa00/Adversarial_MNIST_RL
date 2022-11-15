# Adversarial_MNIST_RL

## Installation

Clone this repo:

```bash
git clone https://github.com/daehwa00/Adversarial_MNIST_RL.git
cd Adversarial_MNIST_RL
```

## Quick Start

### Training

You can train the model with the following command:

```bash
python train.py
```

this command will download the MNIST dataset and train the model.
if you have already downloaded the MNIST dataset, it will not download again.

and this command will save the model in `./results/train/{current_time}` directory.

### options

you can see the option in `train_options.py` file.

#### basic parameters

```bash
parser.add_argument('--dataroot', type=str, default='./MNIST_data',help='path to images')
parser.add_argument('--name', type=str, default='experiment_name',help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--checkpoints_dir', type=str,default='./checkpoints', help='models are saved here')
parser.add_argument('--seed', type=int, default=3, help='random seed')
parser.add_argument('--batch_size', type=int,default=3000, help='input batch size')
parser.add_argument('--use_existing_classification_model', type=bool,default=True, help='use existing classification model')
```

#### model parameters

```bash
parser.add_argument('--cnn_learning_rate', type=float, default=0.005, help='learning rate for cnn')
parser.add_argument('--rl_learning_rate', type=float, default=0.0001, help='learning rate for RL')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')

```

## Prerequistites

- Windows or Linux (This code has not been tested on Mac)
- Python 3.6+
- PyTorch 1.0+
- CUDA 9.0+
- CPU or NVIDIA GPU

## Change Log

2022.11.04:

- Initial release
