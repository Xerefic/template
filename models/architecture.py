from imports import *

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=True)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        
        return x
