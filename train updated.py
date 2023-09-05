import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models

def parse():
    parser=argparse.ArgumentParser(description="Flower Classification model")
    parser.add_argument('--data_dir', type=str, default='flowers', help='directory to get training ,validation and testing data')
    parser.add_argument('--save_dir', type=str, default='./checkpoint.pth', help='directory to save checkpint')
    parser.add_argument('--arch', type=str, default='vgg16', help='model to train',choices=['vgg16', 'resnet18', 'densenet161'])
    parser.add_argument('--epoch', type=int, default='5', help='epoches to train')
    parser.add_argument('--learning_rate', type=float, default='0.001', help='learning rate to train')
    parser.add_argument('--hidden_parameter_in', type=int, default='500', help='Input of hidden layer')
    parser.add_argument('--hidden_parameter_out', type=int, default='200', help='Output of hidden layer')
    
    parser.add_argument('--gpu',default='gpu', action='store_true', help='Use GPU for training if available')
    args=parser.parse_args()
    return args
def preprocess_train(train_dir):
    t=transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])])
    
    train_dataset= datasets.ImageFolder(train_dir, transform=t)
    
    return train_dataset
def preprocess_test(dirr):
    t= transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
    
    valid_dataset= datasets.ImageFolder(dirr, transform=t)
    
    return valid_dataset
def loader(set):
    data_loader = torch.utils.data.DataLoader(set, batch_size=50,shuffle = True)
    
    return data_loader
def my_model(a="vgg16"):
    if a=="vgg16" or a=="VGG16":
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        print("vgg16")
        for param in model.parameters():
            param.requires_grad = False 
    elif a=="densenet" or a=="DENSENET":
        model = models.densenet161(pretrained=True)
        model.name = "densenet161"
        print("densenet")
        
        for param in model.parameters():
            param.requires_grad = False 
            
    elif a=="resnet" or a=="Resnet":
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.name="resnet18"
            
    else:
        print("please select either vgg16,densenet or resnet")
    return model
def architecture(m,inn,out):
    if m.name=="vgg16" :
        in_features_of_pretrained_model = m.classifier[0].in_features
    elif m.name=="resnet18":
        in_features_of_pretrained_model=m.fc.in_features
    elif m.name=="densenet161":
        in_features_of_pretrained_model = m.classifier.in_features
        
    classifier = nn.Sequential(nn.Linear(in_features=in_features_of_pretrained_model, out_features=inn, bias=True),
                           nn.ReLU(inplace=True),
                           nn.Dropout(p=0.2),
                           nn.Linear(in_features=inn, out_features=out, bias=True),
                           nn.ReLU(inplace=True),
                           nn.Linear(in_features=out,out_features= 102),
                           nn.LogSoftmax(dim=1)
                          )
    if m.name=="resnet18":
        m.fc = classifier
    else:
        m.classifier = classifier
    
    
    return m
def training(model,l,train_loader,validate_loader,test_loader,epoch,g,path,train_dataset):
    criterion = nn.NLLLoss()
    if model.name=="resnet18":
         optimizer = optim.Adam(model.fc.parameters(), lr=l)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=l)
    if g and torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif g and not(torch.cuda.is_available()):
        device = 'cpu'
        print("GPU is not available")
    else:
        device = 'cpu'
    print(device)
            
    # move model to selected device
    model.to(device)
    print(model.name)
    for e in range(epoch):
        
        step = 0
        print_every=15
        running_train_loss = 0
        running_valid_loss = 0

        #training
        for images, labels in train_loader:
            
            step += 1
            model.train()
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            train_loss = criterion(outputs, labels)
            
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()
            if step % print_every == 0 or step == 1 or step == len(train_loader):
                print("Epoch: {}/{} training: {:.2f}%".format(e+1, epoch, (step)*100/len(train_loader))) 
        model.eval()
        with torch.no_grad():
            running_accuracy = 0
            running_valid_loss = 0
            for images, labels in validate_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)  
                valid_loss = criterion(outputs, labels)
                running_valid_loss += valid_loss.item()
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        average_train_loss = running_train_loss/len(train_loader)
        average_valid_loss = running_valid_loss/len(validate_loader)
        accuracy = running_accuracy/len(validate_loader)
        print("Train Loss: {:.3f}".format(average_train_loss))
        print("Valid Loss: {:.3f}".format(average_valid_loss))
        print("Valid Accuracy: {:.3f}%".format(accuracy*100)) 
    # TODO: Do validation on the test set
    model.eval()
    with torch.no_grad():
        accuracy = 0
        running_loss = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            quals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        average_loss = running_loss/len(test_loader)
        total_accuracy = accuracy/len(test_loader)
        print("Test Loss: {:.3f}".format(average_loss))
        print("Test Accuracy: {:.3f}".format(total_accuracy))
    model.class_to_idx = train_dataset.class_to_idx
    if model.name=="resnet18":
        a=model.fc
    else:
        a=model.classifier
    
    checkpoint = {'classifier': a,
              'state_dict': model.state_dict(),
                  'm_type': model.name,
              'epochs': epoch,
              'optim_stat_dict': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx
             }
    torch.save(checkpoint, path)

def main():
        args=parse()
        print(torch.__version__)
        data_dir = args.data_dir
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
        train=preprocess_train(train_dir)
        valid=preprocess_test(valid_dir)
        test=preprocess_test(test_dir)
        train_loader=loader(train)
        valid_loader=loader(valid)
        test_loader=loader(test)
        model=my_model(a=args.arch)
        arch=architecture(model,args.hidden_parameter_in,args.hidden_parameter_out)
        t=training(arch,args.learning_rate,train_loader,valid_loader,test_loader,args.epoch,args.gpu,args.save_dir,train)
        
        
        
if __name__ == '__main__': main()
    
    