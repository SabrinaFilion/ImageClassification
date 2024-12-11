import numpy as np
import matplotlib.pyplot as plt
import os
from google.colab import drive
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
import torch.optim as optim
from torchvision import datasets, transforms

#LOADING DATA

drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive/datasets')       #To get the data
# Load data
x = np.load('x_train.npy')  # 11,000 x 64 x 64
y = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

#Converting the training and test sets to tensors
x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)

#Here I am mapping the labels of y to match a zero-indexed mapping
#When submitting my predictions, I will map them back to the actual values representing the result of the equations.
label_mapping = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    12: 9,
    16: 10 }

y_tensor = torch.tensor([label_mapping[label.item()] for label in y])
x_test_tensor_full = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
x_test_tensor=x_test_tensor_full[:4500]   #Halfed version to work with the y_test set

y_test_tensor = torch.tensor(y_test, dtype=torch.long)
y_test_tensor = torch.tensor([label_mapping[label.item()] for label in y_test_tensor])


#DEFINING NETWORK
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Setting up convolution layers
        self.conv1 = nn.Conv2d(1,16,3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32,64,3)
        self.conv4 = nn.Conv2d(64, 128, 3)

        self.pool= nn.MaxPool2d(2,2)  #Using max pooling

        # My Fully connected layers
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 11)      #Using a classification there is a total of
                                          # 11 possible classes (11 possible results of the equations: 1,2,3,4,5,6,7,8,9,12,16) but using zero indexed here

       #Using drop-out to avoid overfitting
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv Layer 1
        x = self.pool(F.relu(self.conv2(x)))  # Conv Layer 2
        x = self.pool(F.relu(self.conv3(x)))  # Conv Layer 3
        x = self.pool(F.relu(self.conv4(x)))  # Conv Layer 4

        x = x.view(-1, 512)

        # Fully connected layers with ReLU activation, as well as drop out to avoid overfitting
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

#Loss function + optimizer
criterion = nn.CrossEntropyLoss()

#using stochastic gradient descent with momentum
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


#DATA AUGMENTATION TRANSFORMATIONS
train_transformations= transforms.Compose([
    transforms.RandomApply([transforms.RandomRotation(20)], p=0.5),
    transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.5)
])

#TRAINING
from torch.utils.data import DataLoader, TensorDataset
train_dataset = TensorDataset(x_tensor, y_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

for epoch in range(65):
    r_loss = 0.0
    for i,(inputs, labels) in enumerate(train_loader, 0):

        inputs = torch.stack([train_transformations(img) for img in inputs])
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward() #compute gradients
        optimizer.step()

        # print statistics
        r_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, r_loss / 100))
            r_loss = 0.0

print('Finished Training')

#EVALUATING MODEL
net.eval()  # Set to evaluation mode
with torch.no_grad():  # No need to compute gradients during evaluation
    test_outputs = net (x_test_tensor)  #  test dataset as a tensor
    test_loss = criterion(test_outputs, y_test_tensor)  # Compare with test labels
    print(f'Test Loss: {test_loss.item()}')










