import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #Define building blocks of CNNs: convolution and pooling layers
        self.conv1 = nn.Conv2d(3, 24, 5, 1)
        self.conv2 = nn.Conv2d(24, 32, 5, 1)
        self.conv3 = nn.Conv2d(32, 50, 5, 1)
        self.pool = nn.MaxPool2d(3, 2)

        #Define fully connected layers
        self.linear1 = nn.Linear(3*3*50, 100)
        self.linear2 = nn.Linear(100, 50)
        self.linear3 = nn.Linear(50, 3)


    def forward(self, x, label):
        """Run forward pass for the network
        :param x: a batch of input images -> Tensor
        :param label: a batch of GT labels -> Tensor
        :return: loss: total loss for the given batch, logits: predicted logits for the given batch
        """

        #Feed a batch of input image x to the main building blocks of CNNs
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)

        #Feed the output of the building blocks of CNNs to the fully connected layers
        x = x.view(-1, 3*3*50)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)

        #Implement cross entropy loss on the top of the output of softmax
        logits = F.softmax(x, dim=1)
        loss = F.cross_entropy(logits, label)

        return loss, logits
