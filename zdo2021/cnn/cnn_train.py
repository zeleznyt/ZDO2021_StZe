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
import torch.nn as nn
import torch.nn.functional as F
import json

from data_loader import VarroaImageDataset 
import varroa_cnn

if __name__ == '__main__':

    batch_size = 4

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((32,32)),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    HOME_PATH = ''

    bg_training_dataset = VarroaImageDataset(train = False, img_dir = HOME_PATH + 'data/bg_train.pkl', transform=transform)
    ob_training_dataset = VarroaImageDataset(train = True, img_dir = HOME_PATH + 'data/ob_train.pkl', transform=transform)

    bg_validation_dataset = VarroaImageDataset(train = False, img_dir = HOME_PATH + 'data/bg_validation.pkl', transform=transform)
    ob_validation_dataset = VarroaImageDataset(train = True, img_dir = HOME_PATH + 'data/ob_validation.pkl', transform=transform)

    classes = ('background', 'varroa')



    train_dataset = torch.utils.data.ConcatDataset([bg_training_dataset,  ob_training_dataset])
    validation_dataset = torch.utils.data.ConcatDataset([bg_validation_dataset, ob_validation_dataset])


    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size,
                                                    shuffle=True, num_workers=2)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                    shuffle=True, num_workers=2)



    def train_cnn(criterion, optimizer, ep):
        history = []
        for epoch in range(ep):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels.type(torch.LongTensor))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                mini = 2000
                if i % mini == mini-1:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / mini))
                    history.append(running_loss / mini)
                    running_loss = 0.0

        print('Finished Training')
        return history



    def AC(loader):
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        with torch.no_grad():
          for data in loader:
              images, labels = data
              outputs = net(images)
              _, predictions = torch.max(outputs, 1)
              # collect the correct predictions for each class
              for label, prediction in zip(labels, predictions):
                  if label == prediction:
                      correct_pred[classes[label]] += 1
                  total_pred[classes[label]] += 1


        # print accuracy for each class
        out = []
        for classname, correct_count in correct_pred.items():

            if( total_pred[classname] == 0):
                print("Accuracy for class {:5s} is: {:.1f} %".format(classname,0))
                out.append((classname, 0))
            else:
                accuracy = 100 * float(correct_count) / total_pred[classname]
                print("Accuracy for class {:5s} is: {:.1f} %".format(classname,accuracy))
                out.append((classname,accuracy))

        return  out


    class_ratio = len(ob_training_dataset) / len(bg_training_dataset)



    print('###################### TRAIN ######################')

    ep = 15
    net = varroa_cnn.Net()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([class_ratio/batch_size, 1.0]))
    history = train_cnn(criterion, optimizer, ep)


    print(net)
    print('Train set')
    ac_train = AC(train_loader)
    print()
    print('Validation set')
    ac_val = AC(validation_loader)

    name = '_01'
    PATH = HOME_PATH + 'log/varoa_net' + name + '.pth'
    torch.save(net.state_dict(), PATH)

    with open(HOME_PATH + 'log/log' + name + '.json', 'w') as f:
        log = [name, history, ac_train, ac_val, ep]
        json.dump(log, f, indent=2)

    print('######################## END ########################')

