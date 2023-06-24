from torchvision import transforms

class Config:
    def __init__(self):
        self.default_initialize()

    def default_initialize(self):
        # 算法核心参数
        self.param_momentum = 0.999
        self.temperature = 0.07
        self.K = 4096
        self.feature_dim = 128
        self.shuffle_BN = True
        
        # 数据增强配置，参考官方代码
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        # 模型训练相关参数
        # Dataloader
        self.batch_size = 256
        self.shuffle = True
        self.num_workers = 32
        # training
        self.train_epoch = 60
        self.optimizer = 'SGD' # now option: ['SGD', ]
        self.init_lr = 0.03
        self.weight_decay = 0.0001
        self.momentum = 0.9
        self.schedular = 'multistepLR' # now option: ['multistepLR']
        self.step = [120, 160]
        # distributed training
        self.train_devices = 8 # change this for distributed training
        self.multiprocess_distributed = True
        self.dist_url = 'tcp://localhost:10001'
        self.dist_backend = 'nccl'
        self.world_size = 1
        self.rank = 0
        # print config
        self.print_freq = 10

    def supervised_settings(self):
        # 数据增强配置
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )
        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_normalize = transforms.Normalize(
            mean=[0.4942, 0.4851, 0.4504], std=[0.2467, 0.2429, 0.2616]
        )
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            val_normalize
        ])

        # 模型训练相关参数
        # Dataloader
        self.batch_size = 128
        self.shuffle = True
        self.num_workers = 32
        # training
        self.train_epoch = 200
        self.optimizer = 'SGD' # now option: ['SGD', ]
        self.init_lr = 0.04
        self.weight_decay = 0.0001
        self.momentum = 0.9
        self.schedular = 'stepLR' # now option: ['multistepLR', 'stepLR']
        self.step = 30
        self.gamma = 0.5
        # distributed training
        self.train_devices = 8 # change this for distributed training
        self.multiprocess_distributed = False
        self.dist_url = 'tcp://localhost:10002'
        self.dist_backend = 'nccl'
        self.world_size = 1
        self.rank = 0
        # print config
        self.print_freq = 1

    def protocol_settings(self):
        self.K = 65536
        # 数据增强配置
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )
        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_normalize = transforms.Normalize(
            mean=[0.4942, 0.4851, 0.4504], std=[0.2467, 0.2429, 0.2616]
        )
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            val_normalize
        ])

        # 模型训练相关参数
        # Dataloader
        self.batch_size = 128
        self.shuffle = True
        self.num_workers = 32
        # training
        self.train_epoch = 50
        self.optimizer = 'SGD' # now option: ['SGD', ]
        self.init_lr = 0.05
        self.weight_decay = 0
        self.momentum = 0.95
        self.schedular = 'stepLR' # now option: ['multistepLR', 'stepLR']
        self.step = 10
        self.gamma = 0.5
        # distributed training
        self.train_devices = 8 # change this for distributed training
        self.multiprocess_distributed = False
        self.dist_url = 'tcp://localhost:10002'
        self.dist_backend = 'nccl'
        self.world_size = 1
        self.rank = 0
        # print config
        self.print_freq = 1


tr_config = Config()
