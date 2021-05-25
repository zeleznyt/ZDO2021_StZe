import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn.functional as F

from . import varroa_cnn
from . import data_loader




def load_net(PATH):
	#PATH = './varoa_net.pth'
	net = varroa_cnn.Net()
	net.load_state_dict(torch.load(PATH))
	net.eval()
	return net
	
def predict(net, image):
	transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((32,32)),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
	#image_set = data_loader.VarroaPredictDataset(train = False, img = image, transform=transform)
	#image_loader = torch.utils.data.DataLoader(image_set, batch_size=1, shuffle=False, num_workers=1)

	with torch.no_grad():
		image = transform(image).unsqueeze(0)
        #empty = torch.Tensor()
        #print(empty)
        #empty = torch.cat([empty, image.detach().clone()])
        #print(empty)
        #print(image)
        #print(empty)
        
		outputs = net(image)
		image.detach()

	
	
	return F.softmax(outputs[0], dim=0)
	#return [1, 0]
