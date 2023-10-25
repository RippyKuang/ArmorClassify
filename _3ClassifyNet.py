"""
@Function:主体网络的搭建

@Time:2022-3-14

@Author:马铭泽
"""
import os
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np

class InvertedResidual(nn.Module):
    def __init__(self,in_channels,out_channels,expand_ratio=9):
        super().__init__()
        hidden_dim=int(in_channels*expand_ratio)# 增大（减小）通道数

        self.conv=nn.Sequential(
                # point wise conv
                nn.Conv2d(in_channels,hidden_dim,kernel_size=1,stride=1,padding=0,groups=1,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # depth wise conv
                nn.Conv2d(hidden_dim,hidden_dim,kernel_size=3,stride=2,padding=1,groups=hidden_dim,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # point wise conv,线性激活（不加ReLU6）
                nn.Conv2d(hidden_dim,out_channels,kernel_size=1,stride=1,padding=0,groups=1,bias=False),
                nn.BatchNorm2d(out_channels)
                )
    def forward(self,x):
        
        return self.conv(x)


class ClassifyNet(nn.Module):
    def __init__(self, In_Channels, Out_Channels,device, file_path, Features=64, fcNodes = 80, LastLayerWithAvgPool = False):
        super(ClassifyNet, self).__init__()
        self.file_path = file_path
        self.device = device
        self.LastLayerWithAvgPool = LastLayerWithAvgPool
        # 特征提取过程
        # 第一层卷积
        self.Conv1 = ClassifyNet.block(In_Channels, Features, "Conv1",Kernel_Size = 5,Padding=2)
        self.Pool1 = nn.AvgPool2d(kernel_size=2, stride=2)#上交采用的是平均池化，但是平均池化保留的是背景，最大保留的是纹理，我不理解为什么用平均
        # 第二层卷积
      
        self.bottleneck1 = InvertedResidual(In_Channels, Features)
      
        # 第三层卷积
        self.bottleneck2 =InvertedResidual(Features *2, Features * 2 ,Features*3)
        self.Conv3 = ClassifyNet.block(Features * 2, Features * 4, "Conv3",Kernel_Size = 3)
        # 拉平
        self.Flatten = nn.Flatten()
        # 全局平均池化
        self.AdaptiveAvgPool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        # 第一层全连接
        self.FC1 = ClassifyNet.fc_block(fcNodes,"fc1")
        # 最后一层分类
        self.FC_End = nn.LazyLinear(Out_Channels)


    # 封装两次卷积过程
    @staticmethod
    def block(In_Channels, Features, name, Kernel_Size, Padding=0):
        """
        :param In_Channels:输入通道数
        :param Features: 第一层卷积后图像的通道数
        :param name: 卷积层的名称
        :param Kernel_Size: 卷积核的大小
        :param Padding: 填充的个数
        :return: 卷积层块(包含卷积层、BN层、Relu)
        """
        return nn.Sequential(OrderedDict([
            # 第一次卷积
            (
                name + "conv",
                nn.Conv2d(In_Channels, Features, kernel_size=Kernel_Size, padding=Padding, bias=True)  # 这里注意可以改是否要加b
            ),
            # batch Normal处理数据分布
            (name + "norm", nn.BatchNorm2d(Features)),
            # Relu激活
            (name + "relu", nn.ReLU(inplace=True)),
            # Gelu激活
            # (name + "Gelu", nn.GELU()),
        ]))

    @staticmethod
    def fc_block(NodesNum, Name, DropRatio=0.5):
        """
        :param NodesNum: 全连接测层的神经元个数
        :param Name: 网络层名称
        :param DropRatio: Dropout的比率
        :return: 全连接层块(包含线性层、Relu、Dropout)
        """
        return nn.Sequential(OrderedDict([
            # 第一次卷积
            (
                Name + "fc",
                nn.LazyLinear(NodesNum)
            ),
            # Relu激活
            (Name + "relu", nn.ReLU(inplace=True)),
            # Gelu激活
            # (Name + "Gelu", nn.GELU()),
            #Dropout
            (Name + "dropout", nn.Dropout(DropRatio)),
        ]))


    @staticmethod
    def flatten(Input,InputShape):
        """
        :param Input: 需要拉成一维向量的输入
        :return: 处理完的一维向量
        """
        OutputShape = InputShape[1] * InputShape[2] * InputShape[3]
        return Input.view(InputShape[0], OutputShape)


    # 正向传播过程
    def forward(self, x):
        """
        :param x: 网络的输入
        :return: 网络的输出
        """
        # 下采样
        self.Conv1.parameters()
        Conv1 = self.Conv1(x)
        bottleneck = self.bottleneck1(x)
        pool1 = self.Pool1(Conv1)
        Conv2 = self.bottleneck2(torch.cat([pool1,bottleneck],dim=1))
        Conv3 = self.Conv3(Conv2)
        # 判断是否使用全局平均池化
        if self.LastLayerWithAvgPool:
            AdaptiveAvgPool = self.AdaptiveAvgPool(Conv3)
        
            FC1 = self.FC1(AdaptiveAvgPool)
            Output = self.FC_End(FC1)
        else:
            Flatten = self.Flatten(Conv3)
            FC1 = self.FC1(Flatten)
            Output = self.FC_End(FC1)

        return Output
# 残差块类
class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)


class HitNet(nn.Module):
    def __init__(self, In_Channels, Out_Channels, Features=64):
        super(HitNet, self).__init__()
        # 特征提取过程
        #第一层卷积
        self.Conv1 = HitNet.block(In_Channels, Features, "Conv1",Kernel_Size = 5,Padding=2)
        #第二层卷积
        self.Conv2 = HitNet.block(Features, Features * 2 , "Conv2",Kernel_Size = 3,Padding=1)
        #残差块
        self.Residual = HitNet.residual(Features * 2, Features * 4, "residual")
        #获取残差块另一边的数据
        self.Conv1x1 = nn.Conv2d(Features * 2, Features * 4, stride=2, kernel_size=1)
        #最后一步的处理
        self.LastBlock = nn.Sequential(nn.ReLU(inplace=True), nn.BatchNorm2d(Features *4), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        #最后一层分类
        self.FC_End = nn.LazyLinear(Out_Channels)


    # 封装两次卷积过程
    @staticmethod
    def block(In_Channels, Features, name,Kernel_Size,Padding = 0):
        """
        :param In_Channels:输入通道数
        :param Features: 第一层卷积后图像的通道数
        :param name: 卷积层的名称
        :param Kernel_Size: 卷积核的大小
        :param Padding: 填充的个数
        :return: 卷积层块(包含卷积层、BN层、Relu)
        """
        return nn.Sequential(OrderedDict([
            # 第一次卷积
            (
                name + "conv",
                nn.Conv2d(In_Channels, Features, kernel_size = Kernel_Size, padding=Padding, bias=True)#这里注意可以改是否要加b
            ),
            # batch Normal处理数据分布
            (name + "norm", nn.BatchNorm2d(Features)),
            # Relu激活
            (name + "relu", nn.ReLU(inplace=True)),
        ]))

    @staticmethod
    def fc_block(NodesNum,Name,DropRatio = 0.5):
        """
        :param NodesNum: 全连接测层的神经元个数
        :param Name: 网络层名称
        :param DropRatio: Dropout的比率
        :return: 全连接层块(包含线性层、Relu、Dropout)
        """
        return nn.Sequential(OrderedDict([
            # 第一次卷积
            (
                Name + "fc",
                nn.LazyLinear(NodesNum)
            ),
            # Relu激活
            (Name + "relu", nn.ReLU(inplace=True)),
            #Dropout
            (Name + "dropout", nn.Dropout(DropRatio)),
        ]))


    @staticmethod
    def flatten(Input,InputShape):
        """
        :param Input: 需要拉成一维向量的输入
        :return: 处理完的一维向量
        """
        OutputShape = InputShape[1] * InputShape[2] * InputShape[3]
        return Input.view(InputShape[0], OutputShape)

    @staticmethod
    def residual(input_channels, num_channels, name):
        return nn.Sequential(OrderedDict([
            # batch Normal处理数据分布
            (name + "norm", nn.BatchNorm2d(input_channels)),
            # Relu激活
            (name + "relu", nn.ReLU(inplace=True)),
            (name + "relu", nn.MaxPool2d(kernel_size=2, stride=2)),
            # 卷积
            (
                name + "conv",
                nn.Conv2d(input_channels, num_channels, kernel_size = 3, padding=1, bias=True)#这里注意可以改是否要加b
            ),
        ]))


    # 正向传播过程
    def forward(self, x):
        """
        :param x: 网络的输入
        :return: 网络的输出
        """
        # 下采样
        Conv1 = self.Conv1(x)
        Conv2 = self.Conv2(Conv1)
        Residual = self.Residual(Conv2)
        Conv1x1 = self.Conv1x1(Conv2)
        Input = Residual + Conv1x1
        FC1 = self.LastBlock(Input)
        Output = self.FC_End(FC1)

        return Output