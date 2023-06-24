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

from models import MoCoRes18
from dataset import ImageNetDatasetForMoCoPretraining
from config import tr_config

def main():
    print("training_epoch:" + str(tr_config.train_epoch))
    print("training_devices:" + str(tr_config.train_devices))
    print("K:" + str(tr_config.K))
    tr_config.distributed = tr_config.world_size > 1 or tr_config.multiprocess_distributed
    ngpus_per_node = tr_config.train_devices
    if (tr_config.batch_size % ngpus_per_node) != 0:
        raise ValueError(
            'You are training with {} gpu, but set batch_size={}'.format(ngpus_per_node, tr_config.batch_size))
    if tr_config.multiprocess_distributed:
        tr_config.world_size = ngpus_per_node * tr_config.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, tr_config))

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")

def main_worker(gpu, ngpus_per_node, config):
    config.gpu = gpu

    if config.multiprocess_distributed and config.gpu != 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    if config.gpu is not None:
        print("Use GPU: {} for training".format(config.gpu))

    if config.distributed:
        if config.dist_url == "env://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocess_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.rank = config.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=config.dist_backend,
            init_method=config.dist_url,
            world_size=config.world_size,
            rank=config.rank,
        )
    
    print("=> Creating model...")
    model = MoCoRes18()

    # distribute the model among GPUs
    if config.distributed:
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            model.cuda(config.gpu)
            config.batch_size = int(config.batch_size / ngpus_per_node)
            config.num_workers = int((config.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[config.gpu]
            )
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    
    cudnn.benchmark = True

    loss_fn = nn.CrossEntropyLoss().cuda(config.gpu)
    
    if tr_config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
                                    model.parameters(), 
                                    config.init_lr, 
                                    momentum=config.momentum, 
                                    weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.step)

    dataset = ImageNetDatasetForMoCoPretraining()

    if config.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=config.batch_size,
                                                shuffle=(sampler is None),
                                                num_workers=config.num_workers,
                                                pin_memory=True,
                                                sampler=sampler,
                                                drop_last=True)
    
    writer = SummaryWriter('./logs/pretraining-K=4096')
    
    # train process
    for epoch in range(config.train_epoch):
        if config.distributed:
            sampler.set_epoch(epoch)

        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")
        progress = ProgressMeter(
            len(dataloader),
            [losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch),
        )

        model.train()

        end = time.time()
        for i, images in enumerate(dataloader):
            data_time.update(time.time() - end)

            if config.gpu is not None:
                images[0] = images[0].cuda(config.gpu, non_blocking=True)
                images[1] = images[1].cuda(config.gpu, non_blocking=True)

            # compute output
            output, target = model(images[0], images[1])
            loss = loss_fn(output, target)

            # acc1/acc5 are (K+1)-way contrast classifier accuracy
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images[0].shape[0])
            top1.update(acc1[0], images[0].shape[0])
            top5.update(acc5[0], images[0].shape[0])

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                progress.display(i)

        writer.add_scalar('loss', losses.avg, epoch)
        writer.add_scalar('top-1 accuracy', top1.avg, epoch)
        writer.add_scalar('top-5 accuracy', top5.avg, epoch)

        scheduler.step()

        if not config.multiprocess_distributed or (
            config.multiprocess_distributed and config.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=False,
                filename="./checkpoint/checkpoint-pretrained-k=4096/checkpoint_{:04d}.pth.tar".format(epoch),
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
        print("\t".join(entries), end="\r")

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
