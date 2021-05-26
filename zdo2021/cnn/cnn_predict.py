import os
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn.functional as F

from . import varroa_cnn
from . import data_loader




def load_net(PATH):
	net = varroa_cnn.Net()
	net.load_state_dict(torch.load(PATH))
	net.eval()
	return net
	
def predict(net, image):
	transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((32,32)),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


	with torch.no_grad():
		image = transform(image).unsqueeze(0)
		outputs = net(image)
		image.detach()

	
	
	return F.softmax(outputs[0], dim=0)
