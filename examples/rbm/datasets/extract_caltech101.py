import numpy as np
from PIL import Image
import glob, os
import ipdb as pdb
import matplotlib.pyplot as plt
import ImageFilter
import ImageOps

newsize=128
data=[]
labels=[]
color=0
jitter_count=5

for label,dir in enumerate(glob.glob("*")):
    num=0
    for file in glob.glob(os.path.join(dir,"*.jpg")):
        if num>=30: break;
        num+=1
        print(file)
        im= Image.open(file)
        if color:
            bw = im.convert("RGB")
        else:
            bw = im.convert("L")
        if bw.size[0] > bw.size[1]:
            bw =bw.resize((newsize,bw.size[1]*newsize /bw.size[0]),Image.BICUBIC)
        else:
            bw =bw.resize((bw.size[0]*newsize /bw.size[1],newsize),Image.BICUBIC)
        bg_color=(np.array(bw.getdata()).mean(axis=0))
        if color:
            bg_color=tuple(bg_color)
            mask=Image.new("RGB",(newsize,newsize),bg_color)
        else:
            mask=Image.new("L",(newsize,newsize),bg_color)
        x_noise= y_noise=0
        for it in xrange(jitter_count):
            x0=(newsize-bw.size[0])/2+x_noise
            y0=(newsize-bw.size[1])/2+y_noise
            x1=x0+bw.size[0]
            y1=y0+bw.size[1]
            x = (x0,y0,x1,y1)
            offset=20
            x_ = (x0+offset,y0+offset,x1+offset,y1+offset)
            mask=ImageOps.expand(mask,offset,fill=bg_color)
            mask.paste(bw,x_)
            for i in xrange(20):
                mask=mask.filter(ImageFilter.SMOOTH)
                mask.paste(bw,x_)
            mask=ImageOps.crop(mask,offset)
            newdata=np.array(mask.getdata()).T.flatten().copy()
            #mask.show()
            #pdb.set_trace()
            #plt.imshow(np.array(mask.getdata()).reshape(newsize,newsize,3))
            #plt.show()
            data.append(newdata)
            labels.append(label)
            x_noise,y_noise=np.random.normal(0,5,2)
            x_noise=max(-20,min(x_noise,20))
            y_noise=max(-20,min(y_noise,20))

if color:
    np.save("caltech_30_color",data)
else:
    np.save("caltech_30_gray",data)
np.save("caltech_labels",labels)
