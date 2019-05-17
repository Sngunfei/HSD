import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):

    def __int__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv1 = nn.Conv2d(1, 6, 5)



def why():

    conv1d = nn.Conv1d(5, 3, 3)
    inputs = torch.ones((2, 3, 2))
    outputs = conv1d(inputs)
    print(outputs)


if __name__ == '__main__':
    why()