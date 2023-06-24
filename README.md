1. ### 项目结构说明

   -dataset	用于存放数据集（或数据集的软连接）

   -logs	用于存放使用Tensorboard保存的训练数据

   -config.py	用于指定训练的参数和细节，可以直接修改

   -dataset.py 	内置 ImageNetDatasetForMoCoPretraining 类，专门用于MoCo方法的数据集读入以及数据增强

   -test_data.ipynb	对于数据集的初步探索，可忽略

   -train.py	自监督训练代码，建议有条件进行多卡训练

   -train-supervised.py	自监督训练微调代码

   ### 模型下载：

   共享文件夹：https://drive.google.com/drive/folders/1Vodx3tNoTpnDwA0bRZ1dW78RKWY7l1m5?usp=sharing

   | ResNet18                       | Google Drive 文件夹内的文件名           |
   | ------------------------------ | --------------------------------------- |
   | 预训练 K=512 Epoch=30          | pretrained_k=512_epoch=0030.pth.tar     |
   | 预训练 K=512 Epoch=40          | pretrained_k=512_epoch=0040.pth.tar     |
   | 预训练 K=4096 Epoch=30         | pretrained_k=4096_epoch=0030.pth.tar    |
   | 预训练 K=4096 Epoch=40         | pretrained_k=4096_epoch=0040.pth.tar    |
   | 预训练 K=65536 Epoch=30        | pretrained_k=65536_epoch=0030.pth.tar   |
   | 预训练 K=65536 Epoch=40        | pretrained_k=65536_epoch=0040.pth.tar   |
   | 预训练 K=65536 Epoch=160       | pretrained_k=65536_epoch=0160.pth.tar   |
   | 预训练 K=65536 Epoch=200       | pretrained_k=65536_epoch=0200.pth.tar   |
   | 监督训练 Epoch=50              | supervised_epoch=0050.pth.tar           |
   | 监督训练 Epoch=200（过拟合）   | supervised_epoch=0200.pth.tar           |
   | 预训练微调 Epoch=50            | pretrained_finetuned_epoch=0050.pth.tar |
   | 预训练微调 Epoch=200（过拟合） | pretrained_finetuned_epoch=0200.pth.tar |

   ### 进行自监督模型训练

   1. 自行准备数据集并重写 torch.utils.data.Dataset 类。如果要使用ImageNet-1M的训练集，可以直接将数据集中的 train 文件夹放至 dataset 文件夹下；
   2. 检查config.py中的参数，调整default_initialize方法中的参数；
   3. 运行train.py即可开始自监督模型训练。

   ### 进行自监督预训练模型的微调

   1. 调整config.py中的参数；
   2. 运行train-supervised.py。

   ### 进行监督训练

   1. 修改train-supervised.py的模型初始化部分后再次运行即可。