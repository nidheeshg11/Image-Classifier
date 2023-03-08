import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json


from functions import load_model,predict,process_image

parser = argparse.ArgumentParser(description='Use neural network to make prediction on image.')

parser.add_argument('input', action='store',default = '',help='Enter path to image.')
parser.add_argument('checkpoint', action='store', default = 'checkpoint.pth', help='Enter location to save checkpoint in.')
parser.add_argument('--top_k', action='store',dest='topk', type=int, default = 3,help='Enter number of top most likely classes to view, default is 3.')
parser.add_argument('--category_names', action='store', dest='cat_name_dir', default = 'cat_to_name.json', help='Enter path to image.')
parser.add_argument('--gpu', action="store", default=False, help='Turn GPU mode on or off, default is off.')
parser.add_argument('--arch', action='store',dest='model', default='vgg13',help='Enter pretrained model to use, default is VGG-13.')

args = parser.parse_args()

checkpoint = args.checkpoint
image = args.input
topk = args.topk
gpu = args.gpu
json_file = args.cat_name_dir
pretrained_m = args.model


with open(json_file, 'r') as f:
    catnames = json.load(f)
    
model = getattr(models,pretrained_m)(pretrained=True)

model = load_model(model, checkpoint, pretrained_m)

image = process_image(image)

if gpu == True:
    image = image.to('cuda')
    
    
probs, classes = predict(image, model, topk, gpu)

names = []

for i in classes:
    names.append(catnames[i])
    
print(f"The top classes from the predictions are :")

for i in range(topk):
    print(f"'{names[i]}' with a probability of {round(probs[i]*100,4)}%")


    

    
