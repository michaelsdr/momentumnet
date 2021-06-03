# Authors: Michael Sander, Pierre Ablin
# License: MIT

import numpy as np
import argparse
from momentumnet.trainer_CIFAR_10 import train_resnet
import os

parser = argparse.ArgumentParser(description="test")
parser.add_argument("-m", default="mresnet18", type=str)
parser.add_argument("-i", default=0, type=int)
parser.add_argument("-s", default="results_cifar_10")
parser.add_argument("-g", default=0.9, type=float)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--save", default=True, type=bool)
args = parser.parse_args()


model = args.m
init_speed = args.i
save_adr = args.s
gamma = args.g
save = args.save
seed = args.seed

if not os.path.isdir("figures"):
    os.mkdir("figures")


n_iters = 220
lr = np.ones(n_iters) * 0.01
lr[180:] /= 10

if __name__ == "__main__":

    train_accs, train_losss, test_accs, test_losss = train_resnet(
        lr, model, init_speed=init_speed, gamma=gamma, seed=seed, save=save
    )

    np.save(
        "%s/%s_%d" % (save_adr, model, seed),
        np.array([train_accs, train_losss, test_accs, test_losss]),
    )
