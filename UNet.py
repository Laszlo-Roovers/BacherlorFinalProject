import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ------------------------- Contractive path -----------------------------------

        # Input Dimension: 256x256x2
        self.enc11 = nn.Conv2d(2, 64, kernel_size=5, padding=2, padding_mode="circular")
        self.enc12 = nn.Conv2d(64, 64, kernel_size=5, padding=2, padding_mode="circular")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

        # Input Dimension: 128x128x64
        self.enc21 = nn.Conv2d(64, 128, kernel_size=5, padding=2, padding_mode="circular")
        self.enc22 = nn.Conv2d(128, 128, kernel_size=5, padding=2, padding_mode="circular")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input Dimension: 64x64x128
        self.enc31 = nn.Conv2d(128, 256, kernel_size=5, padding=2, padding_mode="circular")
        self.enc32 = nn.Conv2d(256, 256, kernel_size=5, padding=2, padding_mode="circular")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input Dimension: 32x32x256
        self.enc41 = nn.Conv2d(256, 512, kernel_size=5, padding=2, padding_mode="circular")
        self.enc42 = nn.Conv2d(512, 512, kernel_size=5, padding=2, padding_mode="circular")

        # ------------------------- Expansive path -----------------------------------

        self.dec11 = nn.Conv2d(512, 256, kernel_size=5, padding=2, padding_mode="circular")
        self.dec12 = nn.Conv2d(256, 256, kernel_size=5, padding=2, padding_mode="circular")
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)

        self.dec21 = nn.Conv2d(256, 128, kernel_size=5, padding=2, padding_mode="circular")
        self.dec22 = nn.Conv2d(128, 128, kernel_size=5, padding=2, padding_mode="circular")
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)

        self.dec31 = nn.Conv2d(128, 64, kernel_size=5, padding=2, padding_mode="circular")
        self.dec32 = nn.Conv2d(64, 64, kernel_size=5, padding=2, padding_mode="circular")
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # ------------------------- Output layer -----------------------------------
        self.outconv = nn.Conv2d(64, 1, kernel_size=5, padding=2, padding_mode="circular")

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

        x_enc11 = nn.functional.relu(self.enc11(x))
        x_enc12 = nn.functional.relu(self.enc12(x_enc11))
        x_p1 = self.pool1(x_enc12)

        x_enc21 = nn.functional.relu(self.enc21(x_p1))
        x_enc22 = nn.functional.relu(self.enc22(x_enc21))
        x_p2 = self.pool2(x_enc22)

        x_enc31 = nn.functional.relu(self.enc31(x_p2))
        x_enc32 = nn.functional.relu(self.enc32(x_enc31))
        x_p3 = self.pool3(x_enc32)

        x_enc41 = nn.functional.relu(self.enc41(x_p3))
        x_enc42 = nn.functional.relu(self.enc42(x_enc41))

        # ------------------------- Decoder -----------------------------------

        x_up1 = self.upconv1(x_enc42)
        x_up11 = torch.cat([x_up1, x_enc32], dim=1)
        x_dec11 = nn.functional.relu(self.dec11(x_up11))
        x_dec12 = nn.functional.relu(self.dec12(x_dec11))

        x_up2 = self.upconv2(x_dec12)
        x_up22 = torch.cat([x_up2, x_enc22], dim=1)
        x_dec21 = nn.functional.relu(self.dec21(x_up22))
        x_dec22 = nn.functional.relu(self.dec22(x_dec21))

        x_up3 = self.upconv3(x_dec22)
        x_up33 = torch.cat([x_up3, x_enc12], dim=1)
        x_dec31 = nn.functional.relu(self.dec31(x_up33))
        x_dec32 = nn.functional.relu(self.dec32(x_dec31))

        # ------------------------- Output -----------------------------------

        out = self.outconv(x_dec32)

        return out
