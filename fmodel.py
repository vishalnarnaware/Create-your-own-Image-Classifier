import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import OrderedDict
import json
from PIL import Image
import futility

def setup_network(structure='vgg16',dropout=0.1,hidden_units=4096, lr=0.001, device='gpu'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    
    for para in model.parameters():
        para.requires_grad = False

    from collections import OrderedDict    

    model.classifier = nn.Sequential(nn.Linear(model.classifier[0].in_features , hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))
    print(model)
    model = model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    if torch.cuda.is_available() and device == 'gpu':
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model.to(device)

    return model, criterion

def save_checkpoint(train_data, model = 0, path = 'checkpoint.pth', structure = 'vgg16', hidden_units = 4096, dropout = 0.3, lr = 0.001, epochs = 1):
    model.class_to_idx =  train_data.class_to_idx
    torch.save({'structure' :structure,
                'hidden_units':hidden_units,
                'dropout':dropout,
                'learning_rate':lr,
                'no_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)
    
def load_checkpoint(path = 'checkpoint.pth'):
    checkpoint = torch.load(path)
    lr = checkpoint['learning_rate']
    hidden_units = checkpoint['hidden_units']
    dropout = checkpoint['dropout']
    epochs = checkpoint['no_of_epochs']
    structure = checkpoint['structure']

    model, _ = setup_network(structure, dropout, hidden_units, lr)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def predict(image_path, model, topk=5, device='gpu'):   
    model.to('cuda')
    model.eval()
    img = process_image(image_path).numpy()
    img = torch.from_numpy(np.array([img])).float()

    with torch.no_grad():
        output = model.forward(img.cuda())
        
    probs = torch.exp(output).data
    
    return probs.topk(topk)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    img_pil = Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    
    image = img_transforms(img_pil)

    return image
