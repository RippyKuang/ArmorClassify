import os.path
import time

import matplotlib.pyplot as plt
from torch import t

from _3ClassifyNet import *
from _2Dataset_Load import *
from _7Evaluate_Model import *
from _8ParametersReader import *
from _0Utility import plot_confusion_matrix
from _1TrainMain import UsingNet

if __name__ == "__main__":
    ClassNum = 9
    device = torch.device("cpu")

    ModelPath = r"./Output/model"
    Model = "1.0_0060.pt"

    FullModelPath = os.path.join(ModelPath, Model)
    Net = BinNet().to(device)  
    Net.load_state_dict(torch.load(FullModelPath, map_location=device))
    Net.eval()

    
    time_start=time.time()
    for x in range(100):
         Input_Tensor = torch.randn([1, 1, 36, 48]).to(device)
         OutputImg = Net(Input_Tensor)
    time_end=time.time()
    print('time cost',time_end-time_start,'s')


#time cost 37.981754541397095 ms