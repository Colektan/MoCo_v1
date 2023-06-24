import torch
from torch import nn
from torchvision.models import resnet18
from config import tr_config

class MoCoRes18(nn.Module):
    '''
    ResNet18 for MoCo Pretraining. Change the tr_config in config.py to customize it.
    '''
    def __init__(self):
        super().__init__()
        self.encoder_q = resnet18(num_classes=tr_config.feature_dim)
        self.encoder_k = resnet18(num_classes=tr_config.feature_dim)

        for q_params, k_params in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            k_params.data.copy_(q_params.data)
            k_params.requires_grad = False

        self.register_buffer("queue", torch.randn(tr_config.feature_dim, tr_config.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        # only one pointer is need because the queue keeps full during the training, the head and the tail are the same.
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    # reference: paper's pseudo code
    def forward(self, img_q, img_k):
        batch = img_q.shape[0]
        dim = tr_config.feature_dim

        # compute feature q
        q = self.encoder_q(img_q) # out: [batch, C]
        q = nn.functional.normalize(q, dim=1)

        # compute feature k
        with torch.no_grad():
            # update encoder_k first
            for q_params, k_params in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                k_params.data.copy_(tr_config.momentum * k_params.data + (1 - tr_config.momentum) * q_params.data)
                                                              
            if tr_config.shuffle_BN:
               img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)
            
            k = self.encoder_k(img_k)
            k = nn.functional.normalize(k, dim=1)

            if tr_config.shuffle_BN:
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # calculate logits
        l_positive = torch.bmm(q.view(batch, 1, dim), k.view(batch, dim, 1)).squeeze(-1) # [batch, 1]
        l_negative = torch.mm(q, self.queue.clone().detach()) # [batch, K]

        # similar to Multi-Classification
        logits = torch.cat((l_positive, l_negative), dim=-1) / tr_config.temperature
        labels = torch.zeros(batch, dtype=torch.long).cuda()

        if tr_config.multiprocess_distributed:
            all_k = concat_all(k)
            all_batch = all_k.shape[0]
        else:
            all_k = k
            all_batch = batch

        ptr = int(self.queue_ptr)

        assert tr_config.K % all_batch == 0

        self.queue[:, ptr : ptr + all_batch] = all_k.T
        self.queue_ptr[0] = (ptr + all_batch) % tr_config.K

        return logits, labels
    
    # shuffle_BN is only available for MutliGPU
    # codes copied from official implementation
    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

@torch.no_grad()
def concat_all(tensor):
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

if __name__=="__main__":
    # from dataset import ImageNetDatasetForMoCoPretraining
    # from torch.utils.data import DataLoader

    # model = MoCoRes18()

    # dataset = ImageNetDatasetForMoCoPretraining()
    # dataloader = DataLoader(dataset, batch_size=32, num_workers=32)
    # tr_config.shuffle_BN = False
    # tr_config.multiprocess_distributed = False

    # for i, data in enumerate(dataloader):
    #     model.forward(data[0], data[1])
    for i in range(100):
        print(i, end='\r')
