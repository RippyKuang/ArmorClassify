from cam_utils import*
from  _3ClassifyNet import*
from PIL import Image
from _2Dataset_Load import*
from torchvision import models, transforms
import numpy as np

if __name__ == "__main__":
    model=BinNet()
   
            
    model.load_state_dict(torch.load("Output/model/1.0_0162.pt"))
    model.eval()

    srcs=[]
    target_label=2
    target_layers = [model.invert_2] 
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    for dirpath, dirnames,filenames in os.walk("./Dataset/Val"):
        for fn in filenames:
            if  int(fn[-5])==target_label:
                srcs.append(Image.open(dirpath+'/'+fn))
    for src in srcs:      
      
        src_tensor = ValImg_Transform(src)
  
        src_tensor = torch.unsqueeze(src_tensor, dim=0) 
        t = model(src_tensor)
        print("Predict Label: "+str(torch.softmax(t, 1, torch.float).argmax(dim=1)))
        grayscale_cam = cam(input_tensor=src_tensor, target=t)


        grayscale_cam = grayscale_cam[0, :]
  
        visualization = show_cam_on_image(np.array(src.convert("RGB")).astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
        plt.imshow(visualization)
        plt.show()