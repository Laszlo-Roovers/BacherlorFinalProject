import torch
import torch.nn as nn
import torchvision
from torchview import draw_graph

from Inception import ConvolutionBlock, InceptionModule

class GoogleNet(nn.Module):
    """Implements the Inceptionv1 architecture.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self, in_channels, num_classes):
        super(GoogleNet, self).__init__()

        self.conv1 = ConvolutionBlock(in_channels, 64, 7, 2, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 =  nn.Sequential(ConvolutionBlock(64,64,1,1,0),ConvolutionBlock(64,192,3,1,1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        # in_channels , out_1x1 , red_3x3 , out_3x3 , red_5x5 , out_5x5 , out_1x1_pooling
        self.inception3a = InceptionModule(192,64,96,128,16,32,32)
        self.inception3b = InceptionModule(256,128,128,192,32,96,64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception4a = InceptionModule(480,192,96,208,16,48,64)
        self.inception4b = InceptionModule(512,160,112,224,24,64,64)
        self.inception4c = InceptionModule(512,128,128,256,24,64,64)
        self.inception4d = InceptionModule(512,112,144,288,32,64,64)
        self.inception4e = InceptionModule(528,256,160,320,32,128,128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception5a = InceptionModule(832,256,160,320,32,128,128)
        self.inception5b = InceptionModule(832,384,192,384,48,128,128)

        self.avgpool = nn.AvgPool2d(kernel_size = 7 , stride = 1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        print("conv1", x.shape)
        x = self.maxpool1(x)
        print("maxpool1", x.shape)

        x = self.conv2(x)
        print("conv2", x.shape)
        x = self.maxpool2(x)
        print("maxpool2", x.shape)

        x = self.inception3a(x)
        print("3a", x.shape)

        x = self.inception3b(x)
        print("3b", x.shape)

        x = self.maxpool3(x)
        print("3bmax", x.shape)

        x = self.inception4a(x)
        print("4a", x.shape)

        x = self.inception4b(x)
        print("4b", x.shape)

        x = self.inception4c(x)
        print("4c", x.shape)

        x = self.inception4d(x)
        print("4d", x.shape)

        x = self.inception4e(x)
        print("4e", x.shape)

        x = self.maxpool4(x)
        print("maxpool", x.shape)

        x = self.inception5a(x)
        print("5a", x.shape)

        x = self.inception5b(x)
        print("5b", x.shape)

        x = self.avgpool(x)
        print("AvgPool", x.shape)

        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        return x

# def testInceptionv1():
x = torch.randn((32, 3, 224, 224))
model = GoogleNet(3, 1000)
# print(model(x).shape)
# return model
# model = testInceptionv1()


architecture = "googlenet"
model_graph = draw_graph(
    model,
    input_size=(1, 3, 224, 224),
    graph_dir="TB",
    roll=True,
    expand_nested=True,
    graph_name=f"self_{architecture}",
    save_graph=True,
    filename=f"self_{architecture}",
)
