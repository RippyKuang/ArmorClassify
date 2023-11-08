"""
@Function:主体网络的搭建

@Time:2022-3-14

@Author:马铭泽
"""
import nntplib
import os
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
class SqueezeExcitation(nn.Module):
    def hardsigmoid(input,inplace=False):
        pass     
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
      
        squeeze_c = input_c // squeeze_factor
      
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x):
      
        scale = nn.AdaptiveAvgPool2d((1, 1))(x)
        scale = self.fc1(scale)
        scale = nn.ReLU( inplace=True)(scale)
        scale = self.fc2(scale)
       
        scale = nn.ReLU(inplace=True)(scale)
        return scale * x        

class InvertedResidual(nn.Module):
    def __init__(self,in_channels,out_channels,expand_ratio=9,_stride=2,is_SE=False):
        super().__init__()
        hidden_dim=int(in_channels*expand_ratio)# 增大（减小）通道数
        self.stride= _stride
        self.conv=nn.Sequential(
                # point wise conv
                nn.Conv2d(in_channels,hidden_dim,kernel_size=1,stride=1,padding=0,groups=1,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # depth wise conv
                nn.Conv2d(hidden_dim,hidden_dim,kernel_size=3,stride=_stride,padding=1,groups=hidden_dim,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # point wise conv,线性激活（不加ReLU6）
                )
        self.pw = nn.Sequential(
                nn.Conv2d(hidden_dim,out_channels,kernel_size=1,stride=1,padding=0,groups=1,bias=False),
                nn.BatchNorm2d(out_channels)
                )
        self.se = SqueezeExcitation(hidden_dim)
        self.is_se = is_SE
        self.is_sc = (in_channels==out_channels)
    def forward(self,x):
        conv_x = self.conv(x)
        if self.is_se ==1:
            conv_x = self.se(conv_x)
        if self.stride==1 and self.is_sc==1:
            return self.pw(conv_x)+x
        else:
            return self.pw(conv_x)
class dw_conv(nn.Module):
    def __init__(self, nin, nout,with_se=False):
        super(dw_conv, self).__init__()
        self.dim=nout
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
        self.with_se = with_se
        self.se = SqueezeExcitation(nout)
        self.bn = nn.BatchNorm2d(self.dim)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        if  self.with_se==True:
            out = self.se(out)
        out = self.bn(out)
        out = nn.ReLU(inplace=True)(out)
        return out

class BinNet(nn.Module):
    def __init__(self):
        super(BinNet, self).__init__()

        self.conv = BinNet.conv_block(1, 4, 3, Padding=1) 
    
        self.invert_1 = InvertedResidual(4,16,1.5,1,True)
       

        self.dw_1 = dw_conv(16,32)
        self.invert_2 = InvertedResidual(32,32,1.5,1,True)
        self.invert_3 = InvertedResidual(32,32,1.5,1,True)


        self.dw_2 = dw_conv(32,48,True)
        self.invert_4 = InvertedResidual(48,48,1.5,1,False)
        self.invert_5 = InvertedResidual(48,48,1.5,2,False)
        self.dw_3 = dw_conv(48,64,True)
        
        self.invert_6 = InvertedResidual(64,64,1.5,1,True)
        self.invert_7 = InvertedResidual(64,64,1.5,2,True)


        self.out_conv1 = nn.Sequential(
                            nn.Conv2d(64, 256, kernel_size=1, stride=1),
                            SqueezeExcitation(256),
                            nn.ReLU(inplace=True),
                        )
        self.FC1 = BinNet.fc_block(60,"fc1")
        # 最后一层分类
        self.FC_End = nn.LazyLinear(9)

    


    def fc_block(NodesNum, Name, DropRatio=0.5):
        return nn.Sequential(
            nn.LazyLinear(NodesNum),
            nn.ReLU(inplace=True),
            nn.Dropout(DropRatio),
        )
    def conv_block(In_Channels, Features,  Kernel_Size, Padding=0):
        return nn.Sequential(
            nn.Conv2d(In_Channels, Features, kernel_size=Kernel_Size, stride=2,padding=Padding, bias=True),
            nn.BatchNorm2d(Features),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x=self.conv(x)
        x=self.invert_1(x)
        x=self.invert_2(self.dw_1(x))
        x=self.invert_3(x)
        x=self.dw_2(x)
        x=self.invert_4(x)
        x=self.invert_5(x)
        x=self.dw_3(x)
        x=self.invert_6(x)
        x=self.invert_7(x)
        x = self.out_conv1(x)
        x = nn.Flatten()(x)
        x = self.FC1(x)
        Output = self.FC_End(x)
        return Output


class ClassifyNet(nn.Module):
    def __init__(self, In_Channels, Out_Channels,device, file_path, Features=64, fcNodes = 60, LastLayerWithAvgPool = False):
        super(ClassifyNet, self).__init__()
        self.file_path = file_path
        self.device = device
        self.LastLayerWithAvgPool = LastLayerWithAvgPool
        # 特征提取过程
        # 第一层卷积
        self.Conv1 = ClassifyNet.block(In_Channels, Features, "Conv1",Kernel_Size = 5,Padding=2)
        self.Pool1 = nn.MaxPool2d(kernel_size=2, stride=2)#上交采用的是平均池化，但是平均池化保留的是背景，最大保留的是纹理，我不理解为什么用平均
        # 第二层卷积
        self.Conv2 = ClassifyNet.block(Features, Features * 2 , "Conv2",Kernel_Size = 3,Padding=1)
        self.Pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第三层卷积
        self.Conv3 = ClassifyNet.block(Features * 2, Features * 4 , "Conv3",Kernel_Size = 3,Padding=1)
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
        Conv2 = self.Conv2(self.Pool1(Conv1))
        Conv3 = self.Conv3(self.Pool2(Conv2))
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

    def get_w(self,fp):
        """
        :param fp: 文件句柄
        :return: 卷积层的权重
        """
        list = fp.readlines()
        for i in range(0, len(list)):
            list[i] = float(list[i].rstrip('\n'))
        weight = torch.Tensor(np.zeros([int(list[1]), int(list[0]), int(list[2]), int(list[3])]))
        index = 0
        for i in range(weight.shape[1]):
            for j in range(weight.shape[0]):
                for k in range(weight.shape[2]):
                    for l in range(weight.shape[3]):
                        weight[j, i, k, l] = list[index + 4]
                        # print(list[index + 4])
                        index += 1

        return weight

    def get_b(self, fp):
        """
        :param fp: 文件句柄
        :return: 卷积层的偏差
        """
        list = fp.readlines()
        for i in range(0, len(list)):
            list[i] = float(list[i].rstrip('\n'))
        bias = torch.Tensor(np.zeros([int(list[0])]))
        for i in range(bias.shape[0]):
            bias[i] = list[i + 1]


        return bias

    def get_fc_w(self,fp):
        """
        :param fp:文件句柄
        :return: 全连接层的权重
        """
        list = fp.readlines()
        for i in range(0, len(list)):
            list[i] = float(list[i].rstrip('\n'))
        weight = torch.Tensor(np.zeros([int(list[1]), int(list[0])])) # (60,560)，nn.Linear乘上的是权重的转置
        index = 0
        for i in range(weight.shape[0]):
            for j in range(weight.shape[1]):
                weight[i, j] = list[index + 2]
                # print(list[index + 4])
                index += 1

        return weight

    def get_fc_b(self, fp):
        """
        :param fp: 文件句柄
        :return: 全连接层的偏差
        """
        list = fp.readlines()
        for i in range(0, len(list)):
            #去除最后的换行符
            list[i] = float(list[i].rstrip('\n'))
        bias = torch.Tensor(np.zeros([int(list[0])]))
        # print("bia shape")
        for i in range(bias.shape[0]):
            bias[i] = list[i + 1]

        return bias


    #权重初始化
    def Para_Init(self):
        """
        :return: None
        """
        for name, parameters in self.named_parameters():
            # 读取卷积层参数
            # conv1
            # print(name.split("."))
            moedl_dict = self.state_dict()
            # print(moedl_dict)
            if (name.split(".")[1] == "Conv1conv1"):#这里读取参数只能通过对中间段进行切片进行，因为中间还有BN层
                if(name.split(".")[2] == "weight"):
                    Full_Path = os.path.join(self.file_path,'conv1_w')
                    with open(Full_Path, mode="r") as fp:
                        parameters.data = torch.nn.Parameter(self.get_w(fp))
                        parameters.requires_grad = False

                if (name.split(".")[2] == "bias"):
                    Full_Path = os.path.join(self.file_path, 'conv1_b')
                    with open(Full_Path, mode="r") as fp:
                        parameters.data = torch.nn.Parameter(self.get_b(fp))
                        parameters.requires_grad = False
                        parameters.data.to(self.device)
                        # print(list(self.Conv1[0].parameters())[0])

            #conv2
            if (name.split(".")[1] == "Conv2conv1"):
                if (name.split(".")[2] == "weight"):
                    Full_Path = os.path.join(self.file_path, 'conv2_w')
                    with open(Full_Path, mode="r") as fp:
                        parameters.data = torch.nn.Parameter(self.get_w(fp))
                        parameters.requires_grad = False
                        parameters.data.to(self.device)
                if (name.split(".")[2] == "bias"):
                    Full_Path = os.path.join(self.file_path, 'conv2_b')
                    with open(Full_Path, mode="r") as fp:
                        parameters.data = torch.nn.Parameter(self.get_b(fp))
                        parameters.requires_grad = False
                        parameters.data.to(self.device)
            #conv3
            if (name.split(".")[1] == "Conv3conv1"):
                if (name.split(".")[2] == "weight"):
                    Full_Path = os.path.join(self.file_path, 'conv3_w')
                    with open(Full_Path, mode="r") as fp:
                        parameters.data = torch.nn.Parameter(self.get_w(fp))
                        parameters.requires_grad = False
                        parameters.data.to(self.device)
                if (name.split(".")[2] == "bias"):
                    Full_Path = os.path.join(self.file_path, 'conv3_b')
                    with open(Full_Path, mode="r") as fp:
                        parameters.data = torch.nn.Parameter(self.get_b(fp))
                        parameters.requires_grad = False
                        parameters.data.to(self.device)

            # fc1
            # if (name.split(".")[1] == "0"):
            #     if (name.split(".")[2] == "weight"):
            #         Full_Path = os.path.join(self.file_path, 'fc1_w')
            #         with open(Full_Path, mode="r") as fp:
            #             parameters.data = torch.nn.Parameter(self.get_fc_w(fp))
            #             parameters.requires_grad = False
            #             parameters.data.to(self.device)
            #     if (name.split(".")[2] == "bias"):
            #         Full_Path = os.path.join(self.file_path, 'fc1_b')
            #         with open(Full_Path, mode="r") as fp:
            #             parameters.data = torch.nn.Parameter(self.get_fc_b(fp))
            #             parameters.requires_grad = False
            #             parameters.data.to(self.device)
            #
            # # fc2
            # if (name.split(".")[1] == "3"):
            #     if (name.split(".")[2] == "weight"):
            #         Full_Path = os.path.join(self.file_path, 'fc2_w')
            #         with open(Full_Path, mode="r") as fp:
            #             parameters.data = torch.nn.Parameter(self.get_fc_w(fp))
            #             parameters.requires_grad = False
            #             parameters.data.to(self.device)
            #     if (name.split(".")[2] == "bias"):
            #         Full_Path = os.path.join(self.file_path, 'fc2_b')
            #         with open(Full_Path, mode="r") as fp:
            #             parameters.data = torch.nn.Parameter(self.get_fc_b(fp))
            #             parameters.requires_grad = False
            #             parameters.data.to(self.device)
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
    

class Two_class(nn.Module):
    def __init__(self, In_Channels, device, file_path, Features = 6, fcNodes = 24):
        # 固定操作
        super(Two_class, self).__init__()
        # 同样，给类成员赋值
        self.file_path = file_path
        self.device = device
        # 特征提取过程
        # 第一层卷积
        self.Conv1 = Two_class.block(In_Channels, Features, "Conv1",Kernel_Size = 5,Padding=2)
        self.Pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 第二层卷积
        self.Conv2 = Two_class.block(Features, Features * 4 , "Conv2",Kernel_Size = 3,Padding=1)
        self.Pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第三层卷积
        self.Conv3 = Two_class.block(Features * 4, Features * 8 , "Conv3",Kernel_Size = 3,Padding=1)
        # 第四层卷积
        self.Conv4 = Two_class.block(Features * 8, Features * 2 , "Conv4",Kernel_Size = 1,Padding=0)
        # 拉平
        self.Flatten = nn.Flatten()
        # 第一层全连接
        self.FC1 = ClassifyNet.fc_block(fcNodes,"fc1")
        self.FC_End = nn.LazyLinear(1)
        self.F = nn.Sigmoid()

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
            # batch Normal处理数据分布（每次处理完图像都标准化）
            (name + "norm", nn.BatchNorm2d(Features)),
            # Relu激活（这个inplace参数填true则不会新建一个图片来返回而是在原有基础上修改，返回原来的地址）
            (name + "relu", nn.ReLU(inplace=True)),
        ]))

    @staticmethod
    def fc_block(NodesNum, Name, DropRatio=0.2):
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
                nn.LazyLinear(NodesNum) # 好好好，懒人线性模块，只需指定输出的维数
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
        # 后三个纬度分别是c*h*w，第一个纬度是batchsize
        OutputShape = InputShape[1] * InputShape[2] * InputShape[3]
        return Input.view(InputShape[0], OutputShape)

    # 正向传播过程
    def forward(self, x):
        """
        :param x: 网络的输入
        :return: 网络的输出
        """
        # 下采样
        self.Conv1.parameters() # 这语句放着干嘛的...
        Conv1 = self.Conv1(x)
        Conv2 = self.Conv2(self.Pool1(Conv1))
        Conv3 = self.Conv3(self.Pool2(Conv2))
        Conv4 = self.Conv4(Conv3)
        Flatten = self.Flatten(Conv4)
        FC1 = self.FC1(Flatten)
        Output = self.F(self.FC_End(FC1))
        return Output