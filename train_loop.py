# we will build MNIST classifier using diffrent components and parameters as follows:
#
# 3-layer fully-connected neural network that takes as input an image that is 28x28 and outputs a probability distribution over 10 possible labels.
#
#
# Optimizer
#   *Stochastic Gradient Descent	https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
#   *Adam Optimizer 	            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
#   *RMSPROP	                    https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop
#
#
# Activation funcation
#   *relu	        https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html#torch.nn.functional.relu
#    hardtanh	    https://pytorch.org/docs/stable/generated/torch.nn.functional.hardtanh.html#torch.nn.functional.hardtanh
#   *softmax	    https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html#torch.nn.functional.softmax
#    tanh	        https://pytorch.org/docs/stable/generated/torch.nn.functional.tanh.html#torch.nn.functional.tanh
#    sigmoid	    https://pytorch.org/docs/stable/generated/torch.nn.functional.sigmoid.html#torch.nn.functional.sigmoid
#    leaky_relu	    https://pytorch.org/docs/stable/generated/torch.nn.functional.leaky_relu.html#torch.nn.functional.leaky_relu

# | lr<br/>SGD,Adam,RMSprop | hidden_units |
# |-------------------------|--------------|
# | 0.005                   | 128          |
# | 0.01                    | 256          |
# | 0.04                    | 512          |
# | 0.08                    | 1024         |
# | 0.1                     | 2048         |
# | 0.2                     |              |
# | 0.5                     |
# | 0.5                     |
#######################################################################################################
#######################################################################################################


import itertools
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from lightning.pytorch.loggers import CSVLogger

from lib.mod3layer import MNISTClassifier, parameterGen

# variables = {
#     "lr": [0.005, 0.01, 0.04, 0.08, 0.1, 0.2, 0.5],
#     "hidden_units": [128, 256, 512, 1024, 2048],
#     "optimizers": [torch.optim.SGD, torch.optim.Adam, torch.optim.RMSprop],
#     "layer_1_act": [torch.sigmoid, torch.relu, torch.tanh],
#     "layer_2_act": [torch.relu, torch.tanh, torch.sigmoid],
#     "layer_3_act": [torch.softmax],
# }

variables = {
    "lr": [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5],
    "hidden_units": [128, 256, 512, 1024, 2048],
    "optimizers": [torch.optim.SGD, torch.optim.Adam, torch.optim.RMSprop],
    "layer_1_act": [torch.relu],
    "layer_2_act": [torch.relu],
    "layer_3_act": [torch.softmax],
}


# Generate combinations
keys = variables.keys()
combinations = [
    dict(zip(keys, combination))
    for combination in itertools.product(*variables.values())
]


print(f"** model count: {len(combinations)}")
# exit(0)

# data
# transforms for images
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


# Print combinations
i = 0
for combination in combinations:
    run_id = f"#{i},"
    run_id += f'lr={combination["lr"]},'
    run_id += f'hidden_units={combination["hidden_units"]},'
    run_id += f'optimizers={combination["optimizers"].__name__},'
    run_id += f'layer_1_act={combination["layer_1_act"].__name__},'
    run_id += f'layer_2_act={combination["layer_2_act"].__name__},'
    run_id += f'layer_3_act={combination["layer_3_act"].__name__}'

    print("\n\n")
    print("#" * 50)
    print(run_id)
    print("#" * 50)

    i += 1

    # prepare transforms standard to MNIST
    mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
    mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)

    train_dataloader = DataLoader(mnist_train, batch_size=64)
    val_loader = DataLoader(mnist_test, batch_size=64)

    v_para = parameterGen(
        lr=combination["lr"],
        hidden_units=combination["hidden_units"],
        optimizers=combination["optimizers"],
        layer_1_act=combination["layer_1_act"],
        layer_2_act=combination["layer_2_act"],
        layer_3_act=combination["layer_3_act"],
    )

    model = MNISTClassifier(v_para)
    logger = CSVLogger("logs", name=run_id)

    trainer = pl.Trainer(min_epochs=1, max_epochs=20, logger=logger)

    trainer.fit(model, train_dataloader, val_loader)

    del mnist_train
    del mnist_test
    del train_dataloader
    del val_loader
    del v_para
    del trainer
    del model
