"""
@Function:主训练程序

@Time:2022-3-14

@Author:马铭泽
"""
import os, logging
from sklearn.multioutput import ClassifierChain
import torch
from _2Dataset_Load import *
from _3ClassifyNet import *
from _0Utility import *
import torch.nn.functional as F
from torchvision import models, transforms
import torchsummary

# 选取要使用的网络
UsingNet = "Net1"  # Net1是全连接神经网络、Net2是带残差层的网络


def train(Epoch, Threshold=False, Ratio=0.6):
    # 开启训练模式，防止Batch Normalization造成的误差
    # Classify.train()
    Net.train()
    # 用来储存损失
    Train_Loss = 0
    # torch.cuda.empty_cache()
    # 显示当前的代数和学习率
    print("Epoch:", Epoch, "Lr=", Lr_Update.get_last_lr()[0], flush=True)

    # 记录计算正确的个数
    TrainCorrect = 0
    Drop_num_Train = 0

    for Num, (InputImg, Label, SampleName) in enumerate(TrainDataLoader):
        # print("第",Num,"次训练",flush=True)
        # 将图像与标签导入GPU
        InputImg = InputImg.float().to(ParaDic.get("Device"))
        Label = Label.to(ParaDic.get("Device"))
        # 权重清零
      
     #   with torch.no_grad():
            #teacher_preds = teacher_Net(InputImg)


        with torch.set_grad_enabled(True):
            torch.cuda.empty_cache()
            if epoch<50:
                a = 0.7
            elif epoch<100:
                a=0.5
            elif epoch<150:
                a=0.3
            elif epoch<200:
                a=0.1
            else:
                a=0
            OutputImg = Net(InputImg)
          
            BatchLoss = Criterion(OutputImg, Label)
           
         #   d_loss = soft_loss(F.log_softmax(OutputImg/temp,dim=1),F.softmax(teacher_preds/temp,dim=1))*temp*temp
          #  loss= a*BatchLoss+(1-a)*d_loss
            Optimizer.zero_grad()
            BatchLoss.backward()
            # 权重更新
            Optimizer.step()
            # 损失的叠加,一定要写item
            Train_Loss += BatchLoss.item()


            # 计算Accuracy
            # 首先找到最大的那个输出，然后和label比较
            Output = torch.softmax(OutputImg, 1, torch.float)   #dim:XXX  A dimension along which softmax will be computed.
  
            # 筛选出准确度高于0.7的样本进行计算
            if Threshold:
                Label = Label[Output.max(dim=1).values > Ratio]
                Drop_num_Train += (Output.max(dim=1).values <= Ratio).sum().item()
                Output = Output[Output.max(dim=1).values > Ratio]

            Predict_ID = Output.argmax(dim=1)
            TrainCorrect += (Predict_ID == Label).sum().item()

    # 计算准确率
    if not Threshold:
        Train_Accuracy = TrainCorrect / (TrainDataset.__len__())
    else:
        Train_Accuracy =TrainCorrect / (TrainDataset.__len__() - Drop_num_Train)

    # 计算平均损失
    Average_Train_Loss = Train_Loss / TrainDataset.__len__() * ParaDic.get("Batchsize")
    print("Train Loss is", Average_Train_Loss, "train Accuracy is", Train_Accuracy, flush=True)
    Log.logger.warning('\tTrain\tEpoch:{0:04d}\tLearningRate:{1:08f}\tLoss:{2:08f}\tAccuracy:{3:08f}\tAlpha:{0:04d}'.format(
        Epoch,
        Lr_Update.get_last_lr()[0],
        Average_Train_Loss,
        Train_Accuracy,a))


def test(Epoch, Threshold=False, Ratio=0.6):
    # Classify.eval()
    Net.eval()
    # 用来储存损失
    Val_Loss = 0
    # 用来存储正确的个数
    ValCorrect = 0
    # 存储不到概率的值
    Drop_num_Val = 0
    # 显示当前的代数和学习率
    print("Epoch:", Epoch, "Lr=", Lr_Update.get_last_lr()[0], flush=True)
    for Num, (InputImg, Label, SampleName) in enumerate(ValDataLoader):
        # 将图像与标签导入GPU
        InputImg = InputImg.float().to(ParaDic.get("Device"))
        Label = Label.to(ParaDic.get("Device"))
        with torch.set_grad_enabled(False):
            # OutputImg = Classify(InputImg)
            OutputImg = Net(InputImg)
            BatchLoss = Criterion(OutputImg, Label)
            # 损失的叠加,一定要写item
            Val_Loss += BatchLoss.item()

            # 判断预测是否正确
            Output = torch.softmax(OutputImg, 1, torch.float)

            # 筛选出准确度高于0.7的样本进行计算
            if Threshold:
                Label = Label[Output.max(dim=1).values > Ratio]
                Drop_num_Val += (Output.max(dim=1).values <= Ratio).sum().item()
                Output = Output[Output.max(dim=1).values > Ratio]

            Predict_ID = Output.argmax(dim=1)
            ValCorrect += (Predict_ID == Label).sum().item()


    # 计算准确率
    if not Threshold:
        Val_Accuracy = ValCorrect / (ValDataset.__len__())
    else:
        Val_Accuracy = ValCorrect / (ValDataset.__len__() - Drop_num_Val)
    # 计算平均损失
    Average_Val_Loss = Val_Loss / ValDataset.__len__()
    print("Test Loss is", Average_Val_Loss, "Test Accuracy is", Val_Accuracy, flush=True)
    Log.logger.warning(
        '\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tValid\tEpoch:{0:04d}\tLearningRate:{1:08f}\tLoss:{2:08f}\tAccuracy:{3:08f}'.format(
        Epoch,
        Lr_Update.get_last_lr()[0],
        Average_Val_Loss,
        Val_Accuracy))




if __name__ == "__main__":
    # 记录版本号
    Version = 1.0
    # 存储网络的相关参数
    ParaDic = {"Epoch": 1000,
               "Lr": 0.0001,# 0.0009
               "Batchsize":32,
               "LrUpdate_Ratio": 0.5,# 0.2
               "LrUpdate_Epoch": 150,
               "TestEpoch": 1,
               "SaveEpoch": 1,
               "Device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
               }

    # 路径读取
    # 数据集路径
    DatasetPath = "Dataset"
    # 模型输出路径
    OutputPath = "Output/model"
    # 日志输出路径
    LogOutputPath = "Output/log"
    os.makedirs(OutputPath, exist_ok=True)
    # 读取数据集
    TrainDataset, TrainDataLoader, ValDataset, ValDataLoader = PipeDatasetLoader(DatasetPath, ParaDic["Batchsize"], False,TrainTransform=TrainImg_Transform, ValTransform=ValImg_Transform)
    temp = 7
    alpha = [0.7,0.5,0.4,0.3,0.1,0.05,0]
    #从此处开始可迭代
    for feature in [6]:
        # 初始化日志系统
        Log = Logger(os.path.join(LogOutputPath, str(Version) + "_" + str(ParaDic.get("Epoch")) + ".log"))


        # SJ网络
        # 实例化对象
        if UsingNet == "Net1":
           #Net = models.mobilenet_v2()
            #print(Net)
            # Net=models.mobilenet_v3_small(pretrained=False)
            # Net.features[0][0] = nn.Conv2d(1 ,32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            # fc_features = Net.classifier[1].in_features
            # Net.classifier[1] = nn.Linear(fc_features, 9) 
           # teacher_Net=BinNet()
           # teacher_Net.load_state_dict(torch.load("Output/model/step3.pt"))
            # Classify.Para_Init()
            
          #  teacher_Net.eval()
           # teacher_Net.to(ParaDic.get("Device"))

            Net = BinNet()
           
            Net.to(ParaDic.get("Device"))

        # 本部网络
        if UsingNet == "Net2":
            Net = HitNet(In_Channels=1, Out_Channels=9,Features=feature)
            Net.to(ParaDic.get("Device"))

       
        torchsummary.summary(Net.cuda(), (1, 36,48))
        Criterion = nn.CrossEntropyLoss().to(ParaDic.get("Device"))
        soft_loss = nn.KLDivLoss(reduction="batchmean")
      
        Optimizer = torch.optim.Adam(Net.parameters(), lr=ParaDic.get("Lr"))
    
        Lr_Update = torch.optim.lr_scheduler.StepLR(Optimizer, ParaDic.get("LrUpdate_Epoch"), gamma=ParaDic.get("LrUpdate_Ratio"))
       
        for epoch in range(1, ParaDic.get("Epoch") + 1):
            train(epoch)
            if epoch % ParaDic.get("TestEpoch") == 0 or epoch == 1:
                test(epoch)

            if epoch % ParaDic.get("SaveEpoch") == 0:
                # torch.save(Classify.state_dict(), os.path.join(OutputPath, '{0}_{1:04d}.pt'.format(Version,epoch)))
                torch.save(Net.state_dict(), os.path.join(OutputPath, '{0}_{1:04d}.pt'.format(Version,epoch)))

            # 学习率的更新
            Lr_Update.step()

        Version += 0.1
        # 保留两位有效数字，防止浮点型精度失真问题
        Version = round(Version, 2)
