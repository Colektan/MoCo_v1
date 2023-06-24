### Logs内容说明

pretraining开头的文件夹表示自监督学习的训练曲线记录，K表示负样本队列长度K的取值。

protocol开头的文件夹表示Linear Classification Protocol的下训练曲线记录，K含义同上。

supervised是监督学习和自监督学习预训练后微调方法的训练曲线记录对比。

在对应的文件夹下运行：

```cmd
tensorboard --logdir ./ --port 6571
```

端口号可自定义。即可展示训练曲线。