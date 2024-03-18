#全连接神经网络

#!/usr/bin/env python
# coding: utf-8

# In[1]:


#载入要用的模块
import torch
import math
import numpy as np
import matplotlib as plt
import torchvision 
from torchvision import datasets,transforms
from torch.autograd import Variable 
from torch import nn
from torch import optim
from matplotlib import pyplot


# In[2]:


#one_hot函数，用于转化实际的y值为一个矩阵
def one_hot(label,depth=10):
    out = torch.zeros(label.size(0),depth)
    idx = torch.LongTensor(label).view(-1,1)
    out.scatter_(dim=1,index=idx,value=1)
    return out

#可视化函数
def plot_curve(data):
    fig = pyplot.figure()
    pyplot.plot(range(len(data)),data,color='yellow')
    pyplot.legend(['value'],loc='upper right')
    pyplot.xlabel('step')
    pyplot.ylabel('value')
    pyplot.show()
def plot_image(img,label,name):
    fig = pyplot.figure()
    for i in range(6):
        pyplot.subplot(2,3,i+1)
        pyplot.tight_layout()
        pyplot.imshow(img[i][0]*0.5+0.5,cmap='gray',interpolation='none')
        pyplot.title("{}: {}".format(name,label[i].item()))
        pyplot.xticks([])
        pyplot.yticks([])
    pyplot.show()
    
#加载数据集
#规定数据集的转换模式
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.5,),(0.5,))])
#用于训练的数据集（直接从网上下载(。・ω・。)）
train_data = torchvision.datasets.MNIST(root="./data/",
                                        transform=transform,
                                        train=True,
                                        download=True)
#用于测试的数据集
test_data = torchvision.datasets.MNIST(root="./data/",
                                       transform=transform,
                                       train=False,
                                       download=True)
#加载
batch = 100
train_data_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch,shuffle=True)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=batch,shuffle=True)

#使手写数字可视化
x,y = next(iter(train_data_loader))
print(x.shape,y.shape,x.min(),x.max())
plot_image(x,y,'sample')


# In[3]:


#创建网络
class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()  #初始化网络
      
        #第一个隐藏层
        self.hidden1 = nn.Sequential(
                                     nn.Linear(28*28,128),
                                     nn.ReLU()             ) 
        
        #第二个隐藏层
        self.hidden2 = nn.Sequential(
                                     nn.Linear(128,256),
                                     nn.ReLU()             )
        #分类层（归一化）
        self.layer = nn.Sequential(
                                     nn.Linear(256,10)
                                     )
        
    def forward(self,x):   #前向传播
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.layer(x)
        return x
    
net = Net()   
net = net.cuda()
optimizer = torch.optim.Adam(net.parameters(),lr=0.001)  #选择优化器
epochs = 3
loss_f = torch.nn.MSELoss()                             #选择损失函数  
loss_f = loss_f.cuda()
train_loss = []

for epoch in range(epochs):                              #梯度下降
    for batch_idx,(x,y) in enumerate(train_data_loader):
        x = x.cuda()
        x = x.view(x.size(0),28*28)
        y_pred = net(x) 
        y_onehot = one_hot(y) 
        y_onehot = y_onehot.cuda()
        loss = loss_f(y_pred,y_onehot)     
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())
        
        if batch_idx % 100 == 0:              #实时显示损失变化
            print(epoch,batch_idx,loss.item())
 
plot_curve(train_loss)                       #画出loss变化曲线图
        


# In[6]:


#评估网络性能
total_correct = 0
for x,y in test_data_loader:
    x = x.cuda()
    y = y.cuda()
    x = x.view(x.size(0),28*28)
    y_pred = net(x)
    pred = y_pred.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct
total_num = len(test_data_loader.dataset)    
acc = total_correct / total_num
print('acc:',acc)


# In[ ]:




