import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """Create patch embeddings from a field of flow data.

    Parameters
    ----------
    img_size : int
        Size of the input field.
    patch_size : int
        Size of a single patch (should be a divisor of `img_size`).
    dim : int
        Dimensionality of the patch embedding.
    channels : int
        Number of channels of the input tensor (2, psi and omega).
    """
    def __init__(self, img_size, patch_size, dim, channels=2):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.num_patches = (img_size // patch_size)**2
        self.patch_dim = channels * patch_size * patch_size

        self.projection = nn.Linear(self.patch_dim, dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, dim))

    def forward(self, x):
        """Calculate a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(Batch, Channel, Heigh, Width)`.

        Returns
        -------
        torch.Tensor
            Patch embeddings of shape `(Batch, num_patches, dim)`.
        """
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, -1, self.patch_dim)
        patches = self.projection(patches)
        patches += self.position_embeddings
        return patches
    
class Upsampler(nn.Module):
    """Upsample feature maps to to match the original input dimensions.

    Parameters
    ----------
    dim : int
        Dimensionality of the feature maps.
    patch_size : int
        Size of a single patch.
    """
    def __init__(self, dim, patch_size):
        super(Upsampler, self).__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim, 1, kernel_size=1)
        )
    
    def forward(self, x):
        """Calculate a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(Batch, dim, Height/patch_size, Width/patch_size).
        
        Returns
        -------
        torch.Tensor
            Upsampled tensor of shape `(Batch, 1, Height, Width)`        
        """
        return self.upsample(x)



class VisionTransformer(nn.Module):
    """Vision Transformer rebuilt to predict turbulent flow.
    
    Parameters
    ----------
    img_size : int
        Size of the input field.
    patch_size : int
        Size of a single patch.
    dim : int
        Dimensionality of the patch embeddings.
    depth : int
        Number of layers in the Transformer encoder.
    heads : int
        Number of attention heads in the encoder.
    mlp_dim : int
        Dimensionality of the feed-forward layers in the Transformer encoder.
    channels : int
        Number of channels in the input tensor (2, psi and omega).
    """
    def __init__(self, img_size, patch_size, dim, depth, heads, mlp_dim, channels=2):
        super(VisionTransformer, self).__init__()

        # Patch embedding and transformer.
        self.patch_embedding = PatchEmbedding(img_size, patch_size, dim, channels)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, mlp_dim), depth
        )
        self.patch_size = patch_size
        self.dim = dim

        # Upsampling to get back to the original dimensions.
        self.upsample = Upsampler(dim, patch_size)

    def forward(self, x):
        """Calculate a forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(Batch, Channel, Heigh, Width)`.
        
        Returns
        -------
        torch.Tensor
            Output field prediction of shape `(Batch, dim, H / patch_size, W / patch_size)`.
        """
        patches = self.patch_embedding(x)
        B, N, D = patches.size()
        x = self.transformer(patches)

        output_size = int(N**0.5)
        compressed_fields = x.view(B, output_size, output_size, self.dim)
        compressed_fields = compressed_fields.permute(0, 3, 1, 2)

        output = self.upsample(compressed_fields)

        return output


model = VisionTransformer(img_size=256, patch_size=16, dim=768, depth=12, heads=12, mlp_dim=3072)
test_input = torch.randn((1, 2, 256, 256))
pred = model(test_input)
print(pred.shape)

from torchview import draw_graph
architecture = 'ViT'
model_graph = draw_graph(
    model,
    input_size=(1, 2, 256, 256),
    graph_dir="TB",
    roll=True,
    expand_nested=True,
    graph_name=f"self_{architecture}",
    save_graph=True,
    filename=f"self_{architecture}",
)
