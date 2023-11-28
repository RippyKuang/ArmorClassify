import os
import cv2 as cv
import random


cnt=0
l= [0,1,1,1,0,0,4,4,5]

for c in ['0','1','2','3','4','5','6','7','8']:
    imgs=[]
    for dirpath,dirnames,filenames in os.walk("./raw"):
        for fn in filenames:
    	    if fn[-5] == c:
    	        img= cv.imread(dirpath+'/'+fn)
    	        imgs.append((cv.cvtColor(img,cv.COLOR_RGB2GRAY),fn))
    random.shuffle(imgs)
    for x in range(0,len(imgs)):
        if x<len(imgs)*0.2:
            cv.imwrite("./aug/Val/"+imgs[x][1],imgs[x][0])
        else:
            cv.imwrite("./aug/Train/"+imgs[x][1],imgs[x][0])
    	    
    print(c)
    		
        

