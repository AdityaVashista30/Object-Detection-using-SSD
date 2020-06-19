# Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

#Defining function of detection
"""a frame, a ssd neural network, and a transformation to be applied on the images
 return the frame with the detector rectangle."""
"""The first transformation to make sure that the image has the right format for torch model.
    convert this transform frame from a number
    Second Transformation: array because it will still be an entire array from a number of array to a torch tensor.
     Third transformation which will be to add a fake dimension to the torch sensor
     fourth and final transformation to do before it is ready to go into the new one that work will be to convert it into a torch variable."""
def detect(frame,net,transform):
    height,width=frame.shape[:2]
    frameT=transform(frame)[0] # transformation of frame
    x = torch.from_numpy(frameT).permute(2, 0, 1) #convert frame into a torch tensor
    #permute: change RBG to GRB for ssd
    x = x.unsqueeze(0)
    with torch.no_grad():
        y = net(x)
    detections = y.data # detections tensor contained in the output y
    #detections=[batch,number of class,number of occurences of each classs,tuple of (score,x0,y0,x1,y1)]
    scale = torch.Tensor([width, height, width, height]) # tensor object of dimensions [width, height, width, height].
    for i in range(detections.size(1)):# For every class:
         j = 0 # will correspond to the occurrences of the class
         while detections[0, i, j, 0] >= 0.6: # taking into account all the occurrences j of the class i that have a matching score larger than 0.6
             pt = (detections[0, i, j, 1:] * scale).numpy() # coordinates of the points at the upper left and the lower right of the detector rectangle.
             cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2) 
             cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA) 
             j += 1 
    return frame

# Creating the SSD neural network
net = build_ssd('test')
# get the weights of the neural network from another one that is pretrained (ssd300_mAP_77.43_v2.pth)
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) 


# Creating the transformation for images to be compatible with ssd neural network
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) #pre specified

# Doing some Object Detection on a video
videoName=str(input("ENTER VIDEO LOCATION TO BE ANALYSED: "))
reader = imageio.get_reader(videoName) 
fps = reader.get_meta_data()['fps'] # get the fps frequence 
writer = imageio.get_writer('output2.mp4', fps = fps) # create output video with this same fps frequence
for i, frame in enumerate(reader): # iterate on the frames of the output video
    frame = detect(frame, net.eval(), transform) 
    writer.append_data(frame) # add the next frame in the output video.
    
print("TOTAL FRAMES PROCESSED: "i) # print the number of the processed frame.
writer.close() 