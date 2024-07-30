import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import cifar100_subset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device '.format(device)) 


#========================Load CIFAR100 testdataset===============================
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

#================================Test Data====================================================
cifar100_subset_testset = cifar100_subset.CIFAR100Subset(
subset=[1, 3, 6, 9, 11, 19, 20, 21, 23, 24, 33, 34, 36, 38, 43, 44, 45, 46, 48, 51, 53, 54, 56, 60, 61, 62, 63, 66, 68, 71],
root='./data/CIFAR100',
train=False,
download=True,
transform=data_transforms["test"])



# summary(cnn_model, (1, 28, 28))
# ====================================================================================
#                    Load Saved ResNet9 Neural network
# ====================================================================================

def test(cnn_model):
    '''Calculates the accuracy of the CNN on the test data'''
    size = len(test_loader.dataset)
    cnn_model.eval()
    with torch.no_grad():
        correct = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            test_output = cnn_model.forward(images)
            pred_y = torch.max(test_output, 1)[1]
            correct += (pred_y == labels).sum()
    accuracy = (correct*100/size) #========Our test data has 18800 images
    print('Test Data Accuracy: {0:.2f}'.format(accuracy))
    return accuracy

#================================Load and test CNN=====================================
if __name__ == '__main__':
    batch_size = 254
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(cifar100_subset_testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    print()
    cnn_model = torch.load('CIFAR100_CNN_trained.json')
    cnn_model.to(device)
    test(cnn_model)

    print('====================== 30 Selected Names of classes ====================================')
    print(cifar100_subset_testset.get_class_names())
    print(len(cifar100_subset_testset))


#=========================================================================================
#     Plotting validation_cost, validation_accuracy, training_cost, training_accuracy
#=========================================================================================
num_epochs = 100 #=================Change this to number of epochs used in training
f = open('CIFAR100_plot_data.json', "r")
[validation_cost, validation_accuracy, training_cost, training_accuracy] = json.load(f)
f.close()

training_cost_xmin = 0
validation_accuracy_xmin = 0
validation_accuracy_xmin = 0
validation_cost_xmin = 0
training_accuracy_xmin = 0
training_set_size = 100000
validation_set_size = 12800

def make_plots(filename, num_epochs, 
               training_cost_xmin, 
               validation_accuracy_xmin, 
               validation_cost_xmin, 
               training_accuracy_xmin,
               training_set_size):
    """Load the results from ``filename``, and generate the corresponding
    plots. """
    f = open(filename, "r")
    validation_cost, validation_accuracy, training_cost, training_accuracy \
        = json.load(f)
    f.close()
    plot_training_cost(training_cost, num_epochs, training_cost_xmin)
    plot_validation_accuracy(validation_accuracy, num_epochs, validation_accuracy_xmin)
    plot_validation_cost(validation_cost, num_epochs, validation_cost_xmin)
    plot_training_accuracy(training_accuracy, num_epochs, 
                           training_accuracy_xmin, training_set_size)

def plot_training_cost(training_cost, num_epochs, training_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_cost_xmin, num_epochs), 
            training_cost[training_cost_xmin:num_epochs],
            color='r', linewidth =2)
    ax.set_xlim([training_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Cost on the CIFAR30 Training Data',fontweight='bold')
    plt.show()

def plot_validation_accuracy(validation_accuracy, num_epochs, validation_accuracy_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(validation_accuracy_xmin, num_epochs), 
            [accuracy
             for accuracy in validation_accuracy[validation_accuracy_xmin:num_epochs]],
            color='g', linewidth =2)
    ax.set_xlim([validation_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy (%) on the CIFAR30 Validation Data',fontweight='bold')
    plt.show()

def plot_validation_cost(validation_cost, num_epochs, validation_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(validation_cost_xmin, num_epochs), 
            validation_cost[validation_cost_xmin:num_epochs],
            color='g', linewidth =2)
    ax.set_xlim([validation_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Cost on the CIFAR30 Validation Data',fontweight='bold')
    plt.show()

def plot_training_accuracy(training_accuracy, num_epochs, 
                           training_accuracy_xmin, training_set_size):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_accuracy_xmin, num_epochs), 
            [accuracy 
             for accuracy in training_accuracy[training_accuracy_xmin:num_epochs]],
            color='r', linewidth =2)
    ax.set_xlim([training_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy (%) on the CIFAR30 Training Data',fontweight='bold')
    plt.show()

make_plots('CIFAR100_plot_data.json', num_epochs, training_cost_xmin, validation_accuracy_xmin,
           validation_cost_xmin,  training_accuracy_xmin, training_set_size)