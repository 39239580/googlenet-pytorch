# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 16:11:47 2018

@author: Weixia
"""

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torch.nn as nn
import time
import torch.nn.functional as F
import os
import csv
# nn.Conv2d是个类，F.conv2d()是个函数，并无区别   类使用大写，

# 定义基本模块
class BasicConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,bias=False,**kwargs)
        self.bn=nn.BatchNorm2d(out_channels,eps=0.001)
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        return F.relu(x,inplace=True)
    
# incption模块
class InceptionModule(nn.Module):
    def __init__(self,in_channels,conv1x1,reduce3x3,conv3x3,reduce5x5,conv5x5,pool_features):
        super().__init__()
        self.branch1x1=BasicConv2d(in_channels,conv1x1,kernel_size=1,stride=1,padding=0)
        
        self.branch3x3_1=BasicConv2d(in_channels,reduce3x3,kernel_size=1,stride=1,padding=0)
        self.branch3x3_2=BasicConv2d(reduce3x3,conv3x3,kernel_size=3,padding=1,stride=1)
        
        self.branch5x5_1=BasicConv2d(in_channels,reduce5x5,kernel_size=1,stride=1,padding=0)
        self.branch5x5_2=BasicConv2d(reduce5x5,conv5x5,kernel_size=5,stride=1,padding=2)
       
        self.branch_pool=BasicConv2d(in_channels,pool_features,kernel_size=1,stride=1,padding=0)
    def forward(self,x):
        branch1x1=self.branch1x1(x)
        
        branch3x3_a=self.branch3x3_1(x)
        branch3x3_b=self.branch3x3_2(branch3x3_a)
        
        branch5x5_a=self.branch5x5_1(x)
        branch5x5_b=self.branch5x5_2(branch5x5_a)

        # 做最大池化，再做卷积
        branch_pool=F.max_pool2d(x,kernel_size=3,stride=1,padding=1)
        branch_pool=self.branch_pool(branch_pool)
        
        outputs=[branch1x1,branch3x3_b,branch5x5_b,branch_pool]
        return torch.cat(outputs,1)
    #torch.cat(inputs,1) 将inputs值按行放在一起，即拼接，即并联
    #torch.cat(inputs,0) 将inputs值按列放在一起，即拼接
# 辅助模块
class InceptionAux(nn.Module):
    def __init__(self,in_channels,out_channels,out,num_classes):
        super().__init()
        self.conv0=BasicConv2d(in_channels,out_channels,kernel_size=1,stride=1)
        self.conv1.stddev=0.01
        self.fc1=nn.Linear(out_channels,out)
        self.fc1.stddev=0.001
        self.fc2=nn.Linear(out,num_classes)
        self.fc2.stddev=0.001
        
    def forward(self,x):
        x=F.avg_pool2d(x,kernel_size=5,stride=3)
        x=self.conv0(x)
        x=x.view(x.size(0),-1) # 将数据转成一列或者一行
        x=self.fc1(x)
        x=self.fc2(x)
        return x
    
# 定义网络结构
class Inception_v1_mnist(nn.Module):
    def __init__(self,num_classes):#,aux_logits_state):
        super().__init__()
        num_class=num_classes
#        self.aux_logits=aux_logits_state
#        self.transform_input=transform_input #输入  28 *28 *1
        self.Conv2d1=BasicConv2d(1,8,kernel_size=5,stride=1,padding=2)#输出 28*28*8
        self.Conv2d2=BasicConv2d(8,32,kernel_size=3,stride=1,padding=1)# 输出 28*28*32

        self.Mixed_3a=InceptionModule(32,16,24,32,4,8,pool_features=8) #输出 28*28*64
        self.Mixed_3b=InceptionModule(64,32,32,48,8,24,pool_features=16) #输出 28*28*120

        self.Mixed_4a=InceptionModule(120,48,24,52,4,12,pool_features=16) #输出 14*14*128
        self.Mixed_4b=InceptionModule(128,40,28,56,6,16,pool_features=16) #输出 14*14*128
        self.Mixed_4c=InceptionModule(128,32,32,64,12,16,pool_features=16) #输出 14*14*128
        self.Mixed_4d=InceptionModule(128,28,36,72,8,16,pool_features=16)  #输出 14*14*132
        self.Mixed_4e=InceptionModule(132,64,40,80,8,32,pool_features=32) #输出 14*14*208
        
        self.Mixed_5a=InceptionModule(208,64,40,80,8,32,pool_features=32) #输出 7*7*208
        self.Mixed_5b=InceptionModule(208,96,48,96,12,32,pool_features=32) ##输出 14*14*256

        #self.fc1=nn.Linear(256,80)  
        self.fc2=nn.Linear(256,num_class)
    def forward(self,transform_input,training_state,dropout_ratio):
        x=transform_input
        dropout_ratio=dropout_ratio
        # 28*28*1
        x = self.Conv2d1(x) 
        x = self.Conv2d2(x)       
        # 28*28*32
        x = self.Mixed_3a(x)
        x = self.Mixed_3b(x)
        #输入28*28*120
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)#输出14*14*120       
        x = self.Mixed_4a(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        #输入14*14*208
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)#输出7*7*208        
        x = self.Mixed_5a(x)
        x = self.Mixed_5b(x)  
        # 输入7*7*256       
        x = F.avg_pool2d(x, kernel_size=7, stride=1, padding=0)#输出1*1*256
        x = F.dropout(x, p=dropout_ratio,training=training_state) #输出1*1*256
        x = x.view(x.size(0), -1) #256
        #x = self.fc1(x)
        x = self.fc2(x)
        return x
def main():
    if os.path.exists('./Inception_v1_mnist_model/model'):
        print('已存在模型文件')
    else:
        print('不存在模型文件')
    print('请输入你的选择：1.训练并测试，2.为直接测试？')
    selection=input()      
    # 超参数
    download_start_time=time.time()
    batch_size=64
    learning_rate=1e-2
    num_epoches=30
    momentum=0.9
    data_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
    #加载数据
    train_dataset=datasets.MNIST(root='./data',train=True,transform=data_tf,download=True)
    test_dataset=datasets.MNIST(root='./data',train=False,transform=data_tf,)
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)   #加载数据
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    download_end_time=time.time()
    print("共计下载数据及加载数据花费时间：%f"%(download_end_time-download_start_time))

    #  对应的实参传入__init__()的四个输入参数中，28*28为图片的大小
    model=Inception_v1_mnist(num_classes=10)  #输出num_classes=10
    model.train()
    if torch.cuda.is_available():
        model=model.cuda()
    else:
        model=model
    print(model)
    criterion=nn.CrossEntropyLoss() #交叉熵作为损失函数
    optimizer=optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum)
    if selection=='1':
        print('进行模型创建并训练')
        train_start_time=time.time()
        train_epoch=[]
        train_loss_value=[]
        train_acc_value=[]
        for epoch in range(num_epoches):
            start=time.time()
            print('Current epoch ={}'.format(epoch)) 
            train_loss=0
            train_acc=0
            for i ,(images,labels)in enumerate(train_loader):#利用enumerate取出一个可迭代对象的内容
                if torch.cuda.is_available():
                    # python中-1为根据系统自动判别个数，
                    inputs=Variable(images).cuda()
                    target=Variable(labels).cuda()
                else:
                    inputs=Variable(images)
                    target=Variable(labels)
                # forward
                out=model(inputs,training_state=True,dropout_ratio=0.4)
                loss=criterion(out,target)
                #backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # loss.item（）是每一轮的损失函数
                train_loss+=loss.item()
                _,pred=torch.max(out,1)
                correct_num=(pred==target).sum()
                train_acc+=correct_num.item()
            train_loss=train_loss/len(train_dataset)
            train_acc=train_acc/len(train_dataset)
            train_epoch.append(epoch+1)
            train_loss_value.append(train_loss)
            train_acc_value.append(train_acc)
            #    if (epoch+1)%10==0:
            print('Epoch[{}/{}],loss:{:.6},acc:{:.6}%,train_time:{:.6}s'.format(epoch+1,num_epoches,train_loss,train_acc*100,time.time()-start))
        train_spend_time=time.time()  
         # 保存模型
        torch.save(model,'./Inception_v1_mnist_model/model')
        print('训练花费时间：%fs'%(train_spend_time-train_start_time))
        with open('./Inception_v1_mnist_model/Inception_v1_mnist_model.csv','w') as csvfile:
            writer=csv.writer(csvfile)
            writer.writerow(["train_epoch","train_acc","train_loss"])          
            for i in range(len(train_epoch)):
                temp=[]
                temp.append(train_epoch[i])
                temp.append(train_acc_value[i])
                temp.append(train_loss_value[i])
                writer.writerow(temp)    
    # 加载模型
    model=torch.load('./Inception_v1_mnist_model/model') 
    # 测试
    model.eval()
    if not torch.cuda.is_available():
        model.cpu()   #模型使用   model.cpu()  
    eval_loss=0
    eval_acc=0
    test_start_time=time.time()
    for data in test_loader:
        img,label=data
        #    img=img.view(img.size(0),-1)
        if torch.cuda.is_available():
            inputs=Variable(img).cuda()
            target=Variable(label).cuda()
        else:
            inputs=Variable(img)
            target=Variable(label)
        out=model(inputs,training_state=False,dropout_ratio=1)
        loss=criterion(out,target)
        eval_loss+=loss.item()
        _,pred=torch.max(out,1)
        num_correct=(pred==target).sum()
        eval_acc+=num_correct.item()
    print('Test Loss:{:.6f},Acc:{:.6}'.format(eval_loss/(len(test_dataset)),100*eval_acc/(len(test_dataset))))
    test_spend_time=time.time()
    print('测试花费时间：%fs'%(test_spend_time-test_start_time))

if __name__=='__main__':
    main()   