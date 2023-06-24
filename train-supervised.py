import builtins
import math
import os
import shutil
import time
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from config import tr_config
from models import MoCoRes18

def main():
    tr_config.K = 512
    tr_config.supervised_settings()
    print("=>training_epoch:" + str(tr_config.train_epoch))
    print("=>training_devices:" + str(tr_config.train_devices))
    tr_config.distributed = tr_config.world_size > 1 or tr_config.multiprocess_distributed
    ngpus_per_node = tr_config.train_devices
    if (tr_config.batch_size % ngpus_per_node) != 0:
        raise ValueError(
            'You are training with {} gpu, but set batch_size={}'.format(ngpus_per_node, tr_config.batch_size))
    if tr_config.multiprocess_distributed:
        tr_config.world_size = ngpus_per_node * tr_config.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, tr_config))
    else:
        main_worker(gpu=None, ngpus_per_node=None, config=tr_config)

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")

def main_worker(gpu, ngpus_per_node, config):
    print("=>Creating model...")
    model_info = torch.load("/sdc1/caiyunuo/pretraining-ResNet/checkpoint/checkpoint-pretrained-k=512/checkpoint_0039.pth.tar")
    weights = model_info['state_dict']

    weights_dict = {}
    for k, v in weights.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    model = resnet18(num_classes=128)
    pretrained_model = MoCoRes18()
    pretrained_model.load_state_dict(weights_dict)
    model.load_state_dict(pretrained_model.encoder_q.state_dict())
    # Replace the last fc for CIFAR-10 task.
    model.fc = torch.nn.Linear(in_features=512, out_features=10)
    model = model.cuda()
    
    cudnn.benchmark = True
    loss_fn = nn.CrossEntropyLoss().cuda()
    
    if tr_config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
                                    model.parameters(), 
                                    config.init_lr, 
                                    momentum=config.momentum, 
                                    weight_decay=config.weight_decay)   
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.step, gamma=config.gamma, verbose=True)

    train_dataset = CIFAR10(root='/sdc1/caiyunuo/pretraining-ResNet/dataset/CIFAR10', train=True, transform=config.augmentation)
    val_dataset = CIFAR10(root='/sdc1/caiyunuo/pretraining-ResNet/dataset/CIFAR10', train=False, transform=config.val_transform)

    sampler = None
    val_sampler = None

    dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=config.batch_size,
                                                shuffle=(sampler is None),
                                                num_workers=config.num_workers,
                                                pin_memory=True,
                                                sampler=sampler,
                                                drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=config.batch_size,
                                                shuffle=(val_sampler is None),
                                                num_workers=config.num_workers,
                                                pin_memory=True,
                                                sampler=val_sampler,
                                                drop_last=True)
    
    writer = SummaryWriter('./logs/supervised')
    
    # train process
    for epoch in range(config.train_epoch):
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        val_losses = AverageMeter("Val Losses", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        val_top1 = AverageMeter("Val Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")
        val_top5 = AverageMeter("Val Acc@5", ":6.2f")
        progress = ProgressMeter(
            len(dataloader),
            [losses, val_losses, top1, val_top1, top5, val_top5],
            prefix="Epoch: [{}]".format(epoch),
        )

        model.train()

        end = time.time()
        print('=>training', end='\r')
        for i, (images, labels) in enumerate(dataloader):
            data_time.update(time.time() - end)

            images = images.cuda()
            labels = labels.cuda()

            # compute output
            output = model(images)
            loss = loss_fn(output, labels)

            # acc1/acc5 are (K+1)-way contrast classifier accuracy
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), images.shape[0])
            top1.update(acc1[0], images.shape[0])
            top5.update(acc5[0], images.shape[0])

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print('=>evaluating', end='\r')
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):

                images = images.cuda()
                labels = labels.cuda()

                # compute output
                output = model(images)
                loss = loss_fn(output, labels)

                # acc1/acc5 are (K+1)-way contrast classifier accuracy
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                val_losses.update(loss.item(), images.shape[0])
                val_top1.update(acc1[0], images.shape[0])
                val_top5.update(acc5[0], images.shape[0])

        writer.add_scalar('train loss', losses.avg, epoch)
        writer.add_scalar('top-1 accuracy', top1.avg, epoch)
        writer.add_scalar('top-5 accuracy', top5.avg, epoch)
        writer.add_scalar('val loss', val_losses.avg, epoch)
        writer.add_scalar('val top-1 accuracy', val_top1.avg, epoch)
        writer.add_scalar('val top-5 accuracy', val_top5.avg, epoch)

        if epoch % config.print_freq == 0:
            progress.display(epoch)

        scheduler.step()

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            is_best=False,
            filename="/sdc1/caiyunuo/pretraining-ResNet/checkpoint/checkpoint-pretrained-supervised/checkpoint_{:04d}.pth.tar".format(epoch),
        )

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
        fmtstr = "{name}:{val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters,  prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.total_iter = num_batches
        self.start_time = time.time()

    def display(self, batch):
        spent_time = time.time() - self.start_time
        remaining_time = spent_time / (batch + 1) * (self.total_iter - batch)
        entries = [self.prefix + self.batch_fmtstr.format(batch)] + ["[%.1f%%"%float(100 * batch / self.total_iter)] + \
                ["%02d:%02d"%divmod(spent_time, 60) +"<" + "%02d:%02d]"%divmod(remaining_time, 60)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == "__main__":
    main()
