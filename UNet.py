import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, name, kernel):
        """Initialize components of the U-Net.

        Parameters
        ----------
        name : str
            Name used for storing results.
        kernel : int
            Makes the convolutional kernel be of size `kernel x kernel`.
        """
        super().__init__()
        self.name = name

        # ------------------------- Contractive path -----------------------------------

        # Input Dimension: 256x256x2
        self.enc11 = nn.Conv2d(2, 16, kernel_size=kernel, padding=2, padding_mode="circular")
        self.bn_enc11 = nn.BatchNorm2d(16)
        self.enc12 = nn.Conv2d(16, 16, kernel_size=kernel, padding=2, padding_mode="circular")
        self.bn_enc12 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

        # Input Dimension: 128x128x64
        self.enc21 = nn.Conv2d(16, 32, kernel_size=kernel, padding=2, padding_mode="circular")
        self.bn_enc21 = nn.BatchNorm2d(32)
        self.enc22 = nn.Conv2d(32, 32, kernel_size=kernel, padding=2, padding_mode="circular")
        self.bn_enc22 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input Dimension: 64x64x128
        self.enc31 = nn.Conv2d(32, 64, kernel_size=kernel, padding=2, padding_mode="circular")
        self.bn_enc31 = nn.BatchNorm2d(64)
        self.enc32 = nn.Conv2d(64, 64, kernel_size=kernel, padding=2, padding_mode="circular")
        self.bn_enc32 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input Dimension: 32x32x256
        self.enc41 = nn.Conv2d(64, 128, kernel_size=kernel, padding=2, padding_mode="circular")
        self.bn_enc41 = nn.BatchNorm2d(128)
        self.enc42 = nn.Conv2d(128, 128, kernel_size=kernel, padding=2, padding_mode="circular")
        self.bn_enc42 = nn.BatchNorm2d(128)

        # ------------------------- Expansive path -----------------------------------

        self.dec11 = nn.Conv2d(128, 64, kernel_size=kernel, padding=2, padding_mode="circular")
        self.bn_dec11 = nn.BatchNorm2d(64)
        self.dec12 = nn.Conv2d(64, 64, kernel_size=kernel, padding=2, padding_mode="circular")
        self.bn_dec12 = nn.BatchNorm2d(64)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.dec21 = nn.Conv2d(64, 32, kernel_size=kernel, padding=2, padding_mode="circular")
        self.bn_dec21 = nn.BatchNorm2d(32)
        self.dec22 = nn.Conv2d(32, 32, kernel_size=kernel, padding=2, padding_mode="circular")
        self.bn_dec22 = nn.BatchNorm2d(32)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.dec31 = nn.Conv2d(32, 16, kernel_size=kernel, padding=2, padding_mode="circular")
        self.bn_dec31 = nn.BatchNorm2d(16)
        self.dec32 = nn.Conv2d(16, 16, kernel_size=kernel, padding=2, padding_mode="circular")
        self.bn_dec32 = nn.BatchNorm2d(16)
        self.upconv3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)

        # ------------------------- Output layer -----------------------------------
        self.outconv = nn.Conv2d(16, 1, kernel_size=kernel, padding=2, padding_mode="circular")

    def forward(self, x):
        """Calculate a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
        """
        # ------------------------- Encoder -----------------------------------

        x_enc11 = nn.functional.relu(self.bn_enc11(self.enc11(x)))
        x_enc12 = nn.functional.relu(self.bn_enc12(self.enc12(x_enc11)))
        x_p1 = self.pool1(x_enc12)

        x_enc21 = nn.functional.relu(self.bn_enc21(self.enc21(x_p1)))
        x_enc22 = nn.functional.relu(self.bn_enc22(self.enc22(x_enc21)))
        x_p2 = self.pool2(x_enc22)

        x_enc31 = nn.functional.relu(self.bn_enc31(self.enc31(x_p2)))
        x_enc32 = nn.functional.relu(self.bn_enc32(self.enc32(x_enc31)))
        x_p3 = self.pool3(x_enc32)

        x_enc41 = nn.functional.relu(self.bn_enc41(self.enc41(x_p3)))
        x_enc42 = nn.functional.relu(self.bn_enc42(self.enc42(x_enc41)))

        # ------------------------- Decoder -----------------------------------

        x_up1 = self.upconv1(x_enc42)
        x_up11 = torch.cat([x_up1, x_enc32], dim=1)
        x_dec11 = nn.functional.relu(self.bn_dec11(self.dec11(x_up11)))
        x_dec12 = nn.functional.relu(self.bn_dec12(self.dec12(x_dec11)))

        x_up2 = self.upconv2(x_dec12)
        x_up22 = torch.cat([x_up2, x_enc22], dim=1)
        x_dec21 = nn.functional.relu(self.bn_dec21(self.dec21(x_up22)))
        x_dec22 = nn.functional.relu(self.bn_dec22(self.dec22(x_dec21)))

        x_up3 = self.upconv3(x_dec22)
        x_up33 = torch.cat([x_up3, x_enc12], dim=1)
        x_dec31 = nn.functional.relu(self.bn_dec31(self.dec31(x_up33)))
        x_dec32 = nn.functional.relu(self.bn_dec32(self.dec32(x_dec31)))

        # ------------------------- Output -----------------------------------

        out = self.outconv(x_dec32)

        return out

from torchview import draw_graph


model = UNet('test', 5)
# print(model(x).shape)
# return model
# model = testInceptionv1()


architecture = "unet"
model_graph = draw_graph(
    model,
    input_size=(16, 2, 256, 256),
    graph_dir="TB",
    roll=True,
    expand_nested=True,
    graph_name=f"self_{architecture}",
    save_graph=True,
    filename=f"self_{architecture}",
)
