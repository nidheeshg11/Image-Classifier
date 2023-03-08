import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import PIL

def loading_data(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    dataloaders = {
        'train':torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid':torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True),
        'test':torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True)
    }
    
    return dataloaders,image_datasets


def classifier_build(model, dropout, input_units, hidden_units, model_name):
    for par in model.parameters():
        par.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1',nn.Linear(input_units, hidden_units)),
        ('relu',nn.ReLU()),
        ('drop',nn.Dropout(p = dropout)),
        ('fc4',nn.Linear(hidden_units,102)),
        ('soft',nn.LogSoftmax(dim = 1))
    ]))
    if model_name.startswith('res'): 
        model.fc = classifier
    elif model_name.startswith('vgg'):
        model.classifier = classifier
    
    return model

def validator(model, loader, crit):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        model.eval()
        accuracy = 0
        for e,(images,labels) in enumerate(loader):
            images,labels = images.to(device),labels.to(device)
            output = model(images)
            test_loss += crit(output,labels)
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.cuda.FloatTensor))
    return test_loss,accuracy



def train_model(model, epochs, trainloader, crit, optim, gpu):
    optimizer = optim
    criterion = crit
    if gpu=='gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    model.to(device)
    for i in range(epochs):
        running_loss = 0
        model.train()
        acc = 0
        for images,labels in trainloader:

            images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output,labels)

            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            r, predict = torch.max(output.data, 1)
            correct_counts = predict.eq(labels.data.view_as(predict))
            acc += torch.mean(correct_counts.type(torch.cuda.FloatTensor))
        else:
            print("Epoch no:{},Running loss:{},Accuracy:{}".format(i+1,running_loss,acc.item()/len(trainloader)))
    return model

def model_eval(model,testloader,criterion,gpu):
    if gpu=='gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    with torch.no_grad():
        model.eval()
        model.to(device)
        accuracy = 0
        for e,(images,labels) in enumerate(testloader):
            images,labels = images.to(device),labels.to(device)
            output = model(images)
            test_loss = criterion(output,labels)
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.cuda.FloatTensor))
            print("Batch no:{},Accuracy:{},Valid loss:{}".format(e+1,accuracy/len(testloader),test_loss.item()))
        
        


def testing(model, testloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.eval()
        accuracy = 0
        for images,labels in testloader:
            images,labels = images.to(device),labels.to(device)
            output = model(images)
            test_loss = criterion(output,labels)
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.cuda.FloatTensor))
            
    return "Accuracy:{}".format(accuracy/len(testloader))


def model_saving(model, input_size, savedir, image_datasets, model_name):
    model.class_to_idx = image_datasets['train'].class_to_idx
    if model_name.startswith('res'):
        checkpoint = {
            'input_size':input_size,
            'state_dict':model.state_dict(),
            'class_to_idx':model.class_to_idx,
            'classifier':model.fc
        }
    elif model_name.startswith('vgg'):
        checkpoint = {
            'input_size':input_size,
            'state_dict':model.state_dict(),
            'class_to_idx':model.class_to_idx,
            'classifier':model.classifier
        }

    return torch.save(checkpoint,savedir)

def load_model(model, savedir, model_name):
    checkpoint = torch.load(savedir)
    model.load_state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    if model_name.startswith('vgg'):
        model.classifier = checkpoint['classifier']
    elif model_name.startswith('res'):
        model.fc = checkpoint['classifier']
    return model

def predict(image, model, topk, gpu):
    if gpu=='gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    model.to(device)
    model.eval()
    torch_image = torch.from_numpy(np.expand_dims(image,axis=0)).type(torch.cuda.FloatTensor)

    output = model.forward(torch_image)
    output = torch.exp(output)
    top_probs, top_labels = output.topk(topk)
    
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    top_labels = [idx_to_class[label] for label in top_labels]   
    
    return top_probs, top_labels



def process_image(image):
   
    pil_image = PIL.Image.open(image).convert('RGB')
   
    
    trans = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    trans_image = trans(pil_image)
    np_image = np.array(trans_image)
    
    return np_image



      