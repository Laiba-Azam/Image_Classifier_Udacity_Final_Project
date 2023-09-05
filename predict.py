import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from torch import tensor
import PIL
import torchvision.models as models
from PIL import Image
from torch.autograd import Variable
from torchvision import  transforms




def parse():
    parser=argparse.ArgumentParser(description="Flower Classification  Inference")
    parser.add_argument('--model_checkpoint', type=str, default='./checkpoint.pth', help='directory to load checkpint')
    parser.add_argument('--category_name', type=str, default='cat_to_name.json', help='directory to load checkpint')
    parser.add_argument('--predict_image_path', type=str, default='flowers/test/9/image_06413.jpg', help='Path Of the image for inference')
    parser.add_argument('--topk', type=int, default=5, help='Classes')
    parser.add_argument('--gpu',default='gpu', action='store_true', help='Use GPU for training if available')
    args=parser.parse_args()
    return args

def loading_the_checkpoint(path):
    #load the saved file
    map_loc=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path,map_location=map_loc)
    #download pretrained model
    m=checkpoint['m_type']
    print(m)
    
    model=models.__dict__[m](pretrained=True)


    #to freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    #load from checkpoint
    if m=="resnet18":
         model.fc = checkpoint['classifier']
    else:
         model.classifier = checkpoint['classifier']
    
   
        
   
    model.class_to_idx = checkpoint['class_to_idx']
    
    model.load_state_dict(checkpoint['state_dict'])
   
    return model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = PIL.Image.open(image)
    preprocess = transforms.Compose([
        transforms.Resize(256),                  # Resize the shortest side to 256 pixels while maintaining aspect ratio
        transforms.CenterCrop(224),              # Crop the center 224x224 portion of the image
        transforms.ToTensor(),                   # Convert to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    
    # Apply the transformations to the image
    img = preprocess(img)
    return img
def predict(image_path, model, topk,cat_to_name):
    # Load the model
    model.eval()
    
    # Preprocess the image
    image = process_image(image_path)
    
    # Ensure the model and image are on the same device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image = image.to(device)
    
    # Add a batch dimension
    image = image.unsqueeze(0)
    
   
    with torch.no_grad():
        output = model(image)
    probabilities = torch.exp(output)
    top_probs, top_indices = probabilities.topk(topk, dim=1)
    
   
    idx_to_class = {v: cat_to_name[k] for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx.item()] for idx in top_indices[0]]
    
    return top_probs[0].tolist(), top_classes


    
        

    

    


def main():
    print(torch.__version__)
    args=parse()
    l=loading_the_checkpoint(args.model_checkpoint)
    with open(args.category_name, 'r') as f:
        cat_to_name = json.load(f)
    probs, classes = predict(args.predict_image_path,l,args.topk,cat_to_name)
    for i in range(args.topk):
        print("{} most pobable class , Probability {:2f}% ,Class name {}".format(i+1,probs[i]*100,classes[i]))
        
    
if __name__ == '__main__': main()