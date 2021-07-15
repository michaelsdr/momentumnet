# Authors: Michael Sander, Pierre Ablin
# License: MIT

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os

from .models import (
    ResNet101,
    mResNet101,
    ResNet18,
    mResNet18,
    mResNet34,
    ResNet34,
    mResNet152,
    ResNet152,
)

n_workers = 10


def train_resnet(
    lr_list,
    model="resnet18",
    use_backprop=False,
    init_speed=0,
    cifar100=False,
    save_adr=None,
    gamma=0.9,
    seed=0,
    save=True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_momnet = model.startswith("m")
    # Data
    expe_name = "ckpt_model_%s_seed_%d_gamma_%.2e.pth" % (model, seed, gamma)
    print("==> Preparing data..")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    if cifar100:
        Loader = torchvision.datasets.CIFAR100
        root = ".data/CIFAR100"
    else:
        Loader = torchvision.datasets.CIFAR10
        root = ".data/CIFAR10"
    trainset = Loader(
        root=root, train=True, download=True, transform=transform_train
    )
    testset = Loader(
        root=root, train=False, download=True, transform=transform_test
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=n_workers
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=n_workers
    )

    # Model
    print("==> Building model..")
    if model == "mresnet18":
        net = mResNet18
    if model == "resnet18":
        net = ResNet18
    if model == "mresnet34":
        net = mResNet34
    if model == "resnet34":
        net = ResNet34
    if model == "mresnet101":
        net = mResNet101
    if model == "resnet101":
        net = ResNet101
    if model == "mresnet152":
        net = mResNet152
    if model == "resnet152":
        net = ResNet152
    num_classes = 100 if cifar100 else 10
    if not is_momnet:
        net = net(num_classes=num_classes)
    else:
        net = net(
            num_classes=num_classes,
            init_speed=init_speed,
            gamma=gamma,
            use_backprop=use_backprop,
        )
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net).cuda()
    resume = os.path.isdir("checkpoint_CIFAR10_resnet")
    if resume:
        assert os.path.isdir(
            "checkpoint_CIFAR10_resnet"
        ), "Error: no checkpoint directory found!"
        try:
            checkpoint = torch.load(
                "./checkpoint_CIFAR10_resnet/%s" % expe_name
            )
            net.load_state_dict(checkpoint["net"])
            print("==> Resuming from checkpoint..")
        except OSError:
            pass

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(
        net.parameters(), lr=lr_list[0], momentum=0.9, weight_decay=5e-4
    )

    # Training
    def train(net, trainloader, epoch):
        print("\nEpoch: %d" % epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_list[epoch]
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(
            "Epoch %d: %.2e, %.2e"
            % (epoch, train_loss / (batch_idx + 1), 100.0 * correct / total)
        )
        return train_loss / (batch_idx + 1), 100.0 * correct / total

    def test(epoch):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print(
            "Test  : %.2e, %.2e"
            % (test_loss / (batch_idx + 1), 100.0 * correct / total)
        )
        return test_loss / (batch_idx + 1), 100.0 * correct / total

    train_accs = []
    train_losss = []
    test_losss = []
    test_accs = []

    for epoch in range(len(lr_list)):
        train_loss, train_acc = train(net, trainloader, epoch)
        test_loss, test_acc = test(epoch)
        train_losss.append(train_loss)
        train_accs.append(train_acc)
        test_losss.append(test_loss)
        test_accs.append(test_acc)
        if save:
            if save_adr is not None:
                np.save(
                    save_adr,
                    np.array([train_accs, train_losss, test_accs, test_losss]),
                )
            state = {
                "net": net.state_dict(),
                "acc": test_acc,
                "epoch": epoch,
            }
            if not os.path.isdir("checkpoint_CIFAR10_resnet"):
                os.mkdir("checkpoint_CIFAR10_resnet")
            torch.save(state, "./checkpoint_CIFAR10_resnet/%s" % expe_name)

    return train_accs, train_losss, test_accs, test_losss
