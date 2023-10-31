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
    if UsingNet == "Net1":
        # TODO:记着换了模型可能要改输出个数
        Net = ClassifyNet(In_Channels=1, Out_Channels=9, Features=6,device=device,file_path=None,LastLayerWithAvgPool=True).to(device)
        Net.load_state_dict(torch.load(os.path.join(FullModelPath, ptModel), map_location=device))

        Net.eval()

    if UsingNet == "Net2":
        # TODO:记着换了模型可能要改输出个数
        Net = HitNet(In_Channels=1, Out_Channels=9, Features=7).to(device)
        # 导入模型
        Net.load_state_dict(torch.load(os.path.join(FullModelPath, ptModel), map_location=device))

        Net.eval()

    # input
    Input_Tensor = torch.randn([1, 1, 36, 48]).to(device)
    input_names = ["input_0"]
    output_names = ["output_0"]
    out_path = onnxName

    # start
    torch.onnx.export(
        Net,
        Input_Tensor,# TODO:这个东西突然有点迷
        out_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={'input_0': [2, 3]}
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
    Create = "test"
    if Create=="False":
        # 创建onnx
        createOnnx('1.0_0001.pt', "avg_this.onnx")
    elif Create =="True":
        # 测试onnx
        OnnxName = "this.onnx"
        TestPath = r"Dataset/Val"

        # 初始化模型
        Model = onnx.load(OnnxName)
        onnx.checker.check_model(Model)
        Ort_session = ort.InferenceSession(OnnxName)


        # 读取图片
        Imgs = os.listdir(TestPath)
        for img in Imgs:
            Img =Image.open(os.path.join(TestPath, img))
            img_temp = Img.copy()
            img_temp = ValImg_Transform(img_temp).view(1,1,36,-1)
            Output = Ort_session.run(None, {'input_0': img_temp.numpy()})
            print("Output", Output[0][0])
            print("Class", np.argmax(Output[0][0]))
            ShowImg = np.array(Img.resize((640, 480)))
            cv.putText(ShowImg, 'Predict:' + str(np.argmax(Output[0][0])), (120, 30), cv.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 0, 0), 1)
            cv.imshow("Img",ShowImg)
            cv.waitKey(0)
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



