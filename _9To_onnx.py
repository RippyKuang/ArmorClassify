"""
@Function: onnx模型的生成与运行

@Time:2022-3-14

@Author:马铭泽
"""

import torch.onnx
import onnx
from _3ClassifyNet import *
from _2Dataset_Load import *
import onnxruntime as ort
from _1TrainMain import UsingNet
import torchsummary

#生成onnx
def createOnnx(ptModel, onnxName):
    """
    :param ptModel: pt模型文件路径
    :param onnxName: onnx模型保存路径
    :return:
    """
    # 选取运算设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 输入模型地址
    FullModelPath = "Output/model"
    # 实例化网络
    Input_Tensor = torch.randn([1, 1, 36, 48]).to(device)
    if UsingNet == "Net1":
        # TODO:记着换了模型可能要改输出个数
       # Net = ClassifyNet(In_Channels=1, Out_Channels=9, Features=6,device=device,file_path=None,LastLayerWithAvgPool=False).to(device)
        Net = BinNet().to(device)
        dict = torch.load(os.path.join(FullModelPath, ptModel), map_location=device)
       
        del dict["FC.0.weight"] 
        del dict["FC.0.bias"]
        del dict["FC_End.weight"]
        del dict["FC_End.bias"]
        torchsummary.summary(Net.cuda(), (1, 36,48))
        Net.load_state_dict(dict)
        
        Net.eval()
        #Net(Input_Tensor)

    if UsingNet == "Net2":
        # TODO:记着换了模型可能要改输出个数
        Net = HitNet(In_Channels=1, Out_Channels=9, Features=7).to(device)
        # 导入模型
        Net.load_state_dict(torch.load(os.path.join(FullModelPath, ptModel), map_location=device))

        Net.eval()

    # input
   
    input_names = ["input_0"]
    output_names = ["output_0"]
    out_path = onnxName

    # start
    torch.onnx.export(
        Net,
        Input_Tensor,# TODO:这个东西突然有点迷
        out_path,
        input_names=input_names,
        output_names=output_names
    )


# 运行并测试onnx
def runOnnx(Input, ModelName):
    """
    :param Input: 用于测试onnx的图像
    :param ModelName: onnx模型名字
    :return: 输出结果
    """
    #run
    Model = onnx.load(ModelName)
    onnx.checker.check_model(Model)
    Ort_session = ort.InferenceSession(ModelName)
    Outputs = Ort_session.run(None,{'input_0':Input})

    return Outputs

if __name__ == "__main__":
    Create = "True"
    if Create=="False":
        # 创建onnx
        createOnnx('step_3.pt', "ksy.onnx")
    elif Create =="True":
        # 测试onnx
        OnnxName = "model.onnx"
        TestPath = r"Dataset/result"

        # 初始化模型
        Model = onnx.load(OnnxName)
        onnx.checker.check_model(Model)
        Ort_session = ort.InferenceSession(OnnxName)

        OutputFolderName = "Output_" + str(0)
        OutputFolderPath = os.path.join("Output/testImg", OutputFolderName)
        os.makedirs(OutputFolderPath, exist_ok=True)

      
        # 读取图片
        Imgs = os.listdir(TestPath)
        for img in Imgs:
            Img =Image.open(os.path.join(TestPath, img))
            img_temp = Img.copy()
            img_temp = ValImg_Transform(img_temp).view(1,1,36,-1)
            Output = Ort_session.run(None, {'input_0': img_temp.numpy()})
            Label = int(os.path.join(TestPath, img).split("/")[2].split(".")[0].split("_")[3])
          
            if Label == np.argmax(Output[0][0]):
                Img.save(OutputFolderPath + r"/{0}".format(img))

           
    else:
        input_data1 = np.random.rand(1, 1, 256, 256).astype(np.float32)
        input_data2 = np.random.rand(1, 1, 512, 512).astype(np.float32)
    
        # 导入 Onnx 模型
        Onnx_file = "./avg_this.onnx"
        Model = onnx.load(Onnx_file)
        onnx.checker.check_model(Model) # 验证Onnx模型是否准确
    
        # 使用 onnxruntime 推理
        model = ort.InferenceSession(Onnx_file)
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
 
        output1 = model.run([output_name], {input_name:input_data1})
        output2 = model.run([output_name], {input_name:input_data2})
 
        print('output1.shape: ', np.squeeze(np.array(output1), 0).shape)
        print('output2.shape: ', np.squeeze(np.array(output2), 0).shape)



