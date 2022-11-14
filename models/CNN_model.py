import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np


"""This code is Shows dotted on the image.
image, label = train_data[0]
shapes = image.squeeze().shape

center = np.random.randint(1,(shapes[0]-2),2)   # between 1 and 26(excluding edge, 28-2)
image = image.squeeze().numpy()
image[center[0],center[1]] = 1
plt.imshow(image, cmap='gray')
"""


# Hyperparameters
batch_size = 3000
learning_rate = 0.005
epoch_num = 30


class CNN(nn.Module):
    def __init__(self, opt):
        super(CNN, self).__init__()
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode, dilation, groups, bias)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="same",
        )
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding="same")
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(3136, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


# classification_model = CNN().to(device)
# optimizer = optim.Adam(classification_model.parameters(), lr=learning_rate)
# criterion = nn.CrossEntropyLoss()


# # Train the CNN model
# classification_model.train()
# for epoch in range(epoch_num):
#     for data, target in train_loader:
#         data = data.to(device)
#         target = target.to(device)
#         optimizer.zero_grad()
#         output = classification_model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#     # epoch마다 loss 출력
#     print('Epoch : {} \tLoss : {:.6f}'.format(epoch, loss.item()))
