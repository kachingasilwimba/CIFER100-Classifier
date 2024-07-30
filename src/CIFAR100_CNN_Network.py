import torch
import torch.nn as nn  # neural network
import torch.nn.functional as F

#==========================================================================================================
#                  CNN Network
#==========================================================================================================
class NetworkCNN_model2(nn.Module):
    def __init__(self, num_classes=30):
        super(NetworkCNN_model2, self).__init__()
        #================ Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        #================ Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        dropout_rate = 0.2
        #================ Fully connected layers with increased dropout rates
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        #================ Flatten the output for the fully connected layers
        x = x.view(-1, 512 * 2 * 2)
        #================ Apply dropout to the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x