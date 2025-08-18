import torch
import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784,256),
                    nn.ReLU(),#增加非线性
                    nn.Linear(256,10))
#初始化权重
def init_weights(m):
    if type(m) == nn.Linear:#因为ReLu不需要初始化权重
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights);
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
#下载数据集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
plt.savefig('trainling_curve.png')
