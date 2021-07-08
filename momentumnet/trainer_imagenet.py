# Authors: Michael Sander, Pierre Ablin
# License: MIT

import os
import datetime
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .models import (
    ResNet18,
    mResNet18,
    ResNet34,
    mResNet34,
    ResNet101,
    mResNet101,
    ResNet152,
    mResNet152,
)


def main(
    datapath,
    saveadr,
    expe_name,
    arch,
    n_workers,
    n_epochs,
    batch_size,
    lr,
    momentum=0.9,
    weight_decay=1e-4,
    print_freq=100,
    init_speed=0,
):

    model_save_adr = "models/%s/%s_checkpoint.pth.tar" % (expe_name, saveadr)
    print("saving model at %s" % model_save_adr)
    resume = os.path.exists(model_save_adr)

    def train(train_loader, model, criterion, optimizer, epoch):
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch),
        )

        # switch to train mode
        model.train()

        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if gpu is not None:
                images = images.cuda(gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)
        return losses.avg, top1.avg.item(), top5.avg.item()

    def validate(val_loader, model, criterion):
        batch_time = AverageMeter("Time", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")
        progress = ProgressMeter(
            len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
        )

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                if gpu is not None:
                    images = images.cuda(gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    progress.display(i)

            print(
                " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(
                    top1=top1, top5=top5
                )
            )

        return losses.avg, top1.avg.item(), top5.avg.item()

    gpu = None
    if gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if gpu is not None:
        print("Use GPU: {} for training".format(gpu))
    # create model
    if arch == "resnet18":
        model = ResNet18(1000)
    if arch == "mresnet18":
        model = mResNet18(1000, init_speed=init_speed)
    if arch == "resnet34":
        model = ResNet34(1000)
    if arch == "mresnet34":
        model = mResNet34(1000, init_speed=init_speed)
    if arch == "resnet101":
        model = ResNet101(1000)
    if arch == "mresnet101":
        model = mResNet101(1000, init_speed=init_speed)
    if arch == "resnet152":
        model = ResNet152(1000)
    if arch == "mresnet152":
        model = mResNet152(1000, init_speed=init_speed)
    if not torch.cuda.is_available():
        print("using CPU, this will be slow")
    if gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    optimizer = torch.optim.SGD(
        model.parameters(), lr, momentum=momentum, weight_decay=weight_decay
    )

    # # optionally resume from a checkpoint
    start_epoch = 0
    train_losss, train_top1s, train_top5s = [], [], []
    test_losss, test_top1s, test_top5s = [], [], []
    if resume:
        print("=> loading checkpoint '{}'".format(model_save_adr))
        if gpu is None:
            checkpoint = torch.load(model_save_adr)
        else:
            # Map model to be loaded to specified single gpu.
            loc = "cuda:{}".format(gpu)
            checkpoint = torch.load(model_save_adr, map_location=loc)
        start_epoch = checkpoint["epoch"]
        print("start epoch : %d" % start_epoch)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_losss = checkpoint["train_loss"]
        train_top1s = checkpoint["train_top1"]
        train_top5s = checkpoint["train_top5"]
        test_losss = checkpoint["test_loss"]
        test_top1s = checkpoint["test_top1"]
        test_top5s = checkpoint["test_top5"]

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(datapath, "train")
    valdir = os.path.join(datapath, "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=n_workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
    )
    print("Start training")
    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()
        adjust_learning_rate(optimizer, epoch, lr)

        # train for one epoch
        train_loss, train_top1, train_top5 = train(
            train_loader, model, criterion, optimizer, epoch
        )

        # evaluate on validation set
        test_loss, test_top1, test_top5 = validate(
            val_loader, model, criterion
        )
        train_losss.append(train_loss)
        train_top1s.append(train_top1)
        train_top5s.append(train_top5)
        test_losss.append(test_loss)
        test_top1s.append(test_top1)
        test_top5s.append(test_top5)
        np.save(
            "results/%s/%s.npy" % (expe_name, saveadr),
            np.array(
                [
                    train_losss,
                    train_top1s,
                    train_top5s,
                    test_losss,
                    test_top1s,
                    test_top5s,
                ]
            ),
        )
        print(
            "Epoch %d took %s"
            % (epoch, str(datetime.timedelta(seconds=int(time.time() - t0))))
        )

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": arch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": train_losss,
                "train_top1": train_top1s,
                "train_top5": train_top5s,
                "test_loss": test_losss,
                "test_top1": test_top1s,
                "test_top5": test_top5s,
            },
            filename=model_save_adr,
        )


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to
    the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top
    predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
