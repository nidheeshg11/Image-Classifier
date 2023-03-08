import torch
from torch import nn
from torch import optim
from torchvision import transforms,datasets,models
from collections import OrderedDict
import PIL
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import argparse
from functions import loading_data,classifier_build,train_model,model_saving,model_eval

    

parser = argparse.ArgumentParser()
parser.add_argument('data_directory', action = 'store',help = 'Enter path to training data.')
parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
parser.add_argument('--dropout', action = 'store', dest='dropout', type=int, default = 0.05, help = 'Enter dropout for training the model, default is 0.05.')
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
parser.add_argument('--gpu', dest="gpu", action="store", default=False)
args = parser.parse_args()


data_dir = args.data_directory
save_dir = args.save_dir
learning_rate = args.learning_rate
dropout = args.dropout
hidden_units = args.hidden_units
epochs = args.epochs
gpu_mode = args.gpu

dataloaders, image_datasets= loading_data(data_dir)


model1 = args.arch
model = getattr(models,model1)(pretrained=True)

if model1.startswith('res'): 
    input_units = model.fc.in_features

elif model1.startswith('vgg'):
    input_units = model.classifier[0].in_features


model = classifier_build(model, dropout, input_units, hidden_units,model1)


if model1.startswith('res'): 
    optimizer = optim.Adam(model.fc.parameters(),float(learning_rate))
elif model1.startswith('vgg'):
    optimizer = optim.Adam(model.classifier.parameters(),float(learning_rate))


criterion = nn.NLLLoss()

model = train_model(model, epochs, dataloaders['train'], criterion, optimizer, gpu_mode)
print("Validation on Model:")

model_eval(model,dataloaders['valid'],criterion,gpu_mode)

model_saving(model, input_units, save_dir, image_datasets,model1)

    
