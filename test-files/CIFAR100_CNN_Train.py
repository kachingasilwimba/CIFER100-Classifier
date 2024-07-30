import sys
sys.path.append(0,'/Users/kachingasilwimba/Desktop/URSSI/CIFER100-Classifier/src')
import torch
from torch import nn
import CIFAR100_CNN_Network
import math
import torch
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import json
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import cifar100_subset
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device '.format(device))


#=======================Load CIFAR100 Dataset================================
batch_size = 164
input_size = 32



data_transforms = {
 'train': transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(32, 4),
        transforms.RandomRotation(15),  # Rotate the image by a random angle
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change the brightness, contrast, saturation and hue
        transforms.RandomVerticalFlip(p=0.5),  # Random vertical flip with a probability of 0.5
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),}
#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

print('====================== 30 Selected Classes ====================================')
cifar100_subset_trainset = cifar100_subset.CIFAR100Subset(
    subset=[1, 3, 6, 9, 11, 19, 20, 21, 23, 24, 33, 34, 36, 38, 43, 44, 45, 46, 48, 51, 53, 54, 56, 60, 61, 62, 63, 66, 68, 71],
    root='/Users/kachingasilwimba/Desktop/URSSI/CIFER100-Classifier/data/CIFAR100',
    train=True,
    download=True,
    transform=data_transforms["train"])

print(cifar100_subset_trainset.get_class_names())
print(len(cifar100_subset_trainset))

#====================== Visualization of 30 Selected Names of the 30 classes ====================================
dataloader = DataLoader(cifar100_subset_trainset, batch_size=64, shuffle=False)
x, _ = next(iter(dataloader))

grid_img = torchvision.utils.make_grid(x, nrow=8)
plt.imshow(grid_img.permute(1, 2, 0))
plt.savefig('Selected_30_Classes.pdf', dpi=300)
# plt.show()

#====================== Split the filtered training dataset into training and validation sets ======================
train_size = int(0.8 * len(cifar100_subset_trainset))
validation_size = len(cifar100_subset_trainset) - train_size
train_data, validation_data = torch.utils.data.random_split(cifar100_subset_trainset, [train_size, validation_size])

#==========================CIFAR100 Data Description============================
print()
print('=============CIFAR100 Validation Data Shape and Type ====================')
print('type(validation_data) =', type(validation_data))
print('len(validation_data) =', len(validation_data))
print('type(validation_data[0]) =', type(validation_data[0]))
print('type(validation_data[1]) =', type(validation_data[1]))
print('len(validation_data[0]) =', len(validation_data[0]))
print('validation_data[0][0].shape = ', validation_data[0][0].shape)
print('validation_data[0][1] =', validation_data[0][1])
img,label = validation_data[0]
print('img.shape =', img.shape)
print('label =',label) 
print()

print('=============CIFAR100 Training Data Shape and Type ========================')
print('type(training_data) =', type(train_data))
print('len(training_data) =', len(train_data))
print('type(training_data[0]) =', type(train_data[0]))
print('type(training_data[1]) =', type(train_data[1]))
print('len(training_data[0]) =', len(train_data[0]))
print('training_data[0][0].shape = ', train_data[0][0].shape)
print('training_data[0][1] =', train_data[0][1])
img,label = train_data[0]
print('img.shape =', img.shape)
print('label =',label)
print()

#====================================================================================
#                    Tranining Convolution Neural network
#====================================================================================

def train_loop(cnn_model, optimizer, loss_fn, batch_size):
    '''
    Returns validation loss and accuracy
        Parameters:
            cnn_model (CNN): a convolutional neural network to train
            optimizer: optimizer
             loss function: a loss function to evaluate the model on 
        Returns:
            cnn_model (CNN): a trained model
            train_loss (float): train loss
            train_acc (float): train accuracy
    '''
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    cnn_model.train()
    correct = 0
    total = 0
    train_loss = 0
    
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = cnn_model(inputs)
        
        optimizer.zero_grad()
        
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        #===================== the class with the highest value is the prediction
        _, prediction = torch.max(outputs.data, 1)  #====== grab prediction as one-dimensional tensor
        total += labels.size(0)
        correct += (prediction == labels).sum().item()

    training_loss = train_loss/len(train_loader)
    training_accs = 100*correct/total
    return cnn_model, training_loss, training_accs  

#====================================================================================
#                    Validation of Convolution Neural network
#====================================================================================

def valid_loop(cnn_model, loss_fn ,batch_size):
    '''
    Returns validation loss and accuracy
    
        Parameters:
            cnn_model (CNN): a convolutional neural network to validate
            loss function: a loss function to evaluate the model on
        
        Returns:
            validation_loss (float): validation loss
            validation_accs (float): validation accuracy
    '''
    
    val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)   
    cnn_model.eval()
    correct = 0
    total = 0
    val_loss = 0 
    
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = cnn_model(inputs)

            loss = loss_fn(outputs, labels)
            
            val_loss += loss.item()
            _, prediction = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
            
        validation_loss = val_loss/len(val_loader)
        validation_accs = 100*correct/total

    return validation_accs, validation_loss


#====================================================================================
#                    Tranining and Save Residual Neural network
#====================================================================================

def train_save():   
    '''
    Execute train and validate functions epoch-times to train a CNN model.
    Each time, store train & validation loss and accuracy.
    Then, test the model and return the result. ResNetModified
    '''


    #=====================CNN Model
    NetworkCNN_model2 = CIFAR100_CNN_Network.NetworkCNN_model2(num_classes=30)#.to(device) #Your model
   
    cnn_model = NetworkCNN_model2
    #=====================loss function
    loss_fn = nn.CrossEntropyLoss()
    #=====================optimizer
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr = 0.0001)  # 0.01 learning rate
    
    num_epochs = 150 #=====================number of training to be completed
    batch_size = 260 #254
    
    #=====================containers to keep track of statistics
    training_cost = []
    validation_cost = []
    training_accuracy = []
    validation_accuracy  = []
       
    for epoch in range(num_epochs):  #=====================number of training to be completed
        cnn_model, training_loss, training_accs = train_loop(cnn_model, optimizer, loss_fn, batch_size)
        validation_accs, validation_loss = valid_loop(cnn_model, loss_fn, batch_size)
        
        training_cost.append(training_loss)
        validation_cost.append(validation_loss)
        training_accuracy.append(training_accs)
        validation_accuracy.append(validation_accs)
        
    #=====================print results of each iteration
        print(f'Epoch [{epoch+1}/{num_epochs}]=======================================================================')
        print(f'Accuracy(train, validation):{round(training_accs,1),round(validation_accs,1)}%, Loss(train,validation):{round(training_loss,4), round(validation_loss,4)}')
        print()
    torch.save(cnn_model, '/Users/kachingasilwimba/Desktop/URSSI/CIFER100-Classifier/saved_model/CIFAR100_CNN_trained.json') 
    filename = '/Users/kachingasilwimba/Desktop/URSSI/CIFER100-Classifier/saved_model/CIFAR100_plot_data.json'
    f = open(filename, "w")
    json.dump([validation_cost, validation_accuracy, training_cost, training_accuracy], f)
    f.close() 
train_save()
