from cam_utils import*
from  _3ClassifyNet import*
from PIL import Image
from _2Dataset_Load import*
from torchvision import models, transforms
import numpy as np

if __name__ == "__main__":
    model=BinNet()
   
            
    model.load_state_dict(torch.load("Output/model/step_3.pt"))

    model.eval()
    src = Image.open("./error3.bmp") #args.img_src是预设的图片路径
    
    src_tensor = ValImg_Transform(src)
    print(src_tensor.shape)
    src_tensor = torch.unsqueeze(src_tensor, dim=0) 
#这里是因为模型接受的数据维度是[B,C,H,W]，输入的只有一张图片所以需要升维

#3）指定需要计算CAM的网络结构
    target_layers = [model.invert_2] #down4()是在Net网络中__init__()方法中定义了的self.down4

#5）可视化展示结果
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=src_tensor, target=model(src_tensor))


    grayscale_cam = grayscale_cam[0, :]
  
    visualization = show_cam_on_image(np.array(src.convert("RGB")).astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()