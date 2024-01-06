import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #####Insert your code here for subtask 1d#####

        #Define building blocks of CNNs: convolution and pooling layers

        #####Insert your code here for subtask 1e#####

        #Define fully connected layers

    def forward(self, x, label):
        """Run forward pass for the network
        :param x: a batch of input images -> Tensor
        :param label: a batch of GT labels -> Tensor
        :return: loss: total loss for the given batch, logits: predicted logits for the given batch
        """

        #####Insert your code here for subtask 1f#####
        #Feed a batch of input image x to the main building blocks of CNNs
        #Do not forget to implement ReLU activation layers here

        #####Insert your code here for subtask 1g#####
        #Feed the output of the building blocks of CNNs to the fully connected layers

        #####Insert your code here for subtask 1h#####
        #Implement cross entropy loss on the top of the output of softmax
        logits = F.softmax(x, dim=1)

        return loss, logits
