import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import List


class AudioMLP(nn.Module):
    """Simple feed-forward model for flat MFCC vectors."""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, input_dim)
        return self.net(x)
class ResidualBlock(nn.Module):
    """Residual block for audio classification"""
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
    def forward(self, x):
        residual = self.skip(x)
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        out += residual  # Residual connection
        out = F.relu(out)
        
        return out

class AudioResNet(nn.Module):
    """ResNet-style architecture for audio feature classification"""
    def __init__(self, input_dim: int, num_classes: int, hidden_dims=[128, 256, 256, 128], dropout=0.3):
        super().__init__()
        
        # Initial projection
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.bn_input = nn.BatchNorm1d(hidden_dims[0])
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.res_blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i+1], dropout=dropout)
            )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Initial projection
        x = self.input_layer(x)
        x = self.bn_input(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Output
        x = self.output_layer(x)
        
        return x







class DepthwiseSeparableConv2d(nn.Module):
    """MobileNetV1 depthwise-separable block used by YAMNet-style backbones."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.bn_depthwise = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_pointwise = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn_depthwise(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.bn_pointwise(x)
        x = self.act(x)
        return x


class FineTunableYAMNet(nn.Module):
    """Fine-tunable model aligned with the AudioResNet training scheme.

    This keeps the same class entry point but uses a residual MLP backbone so
    it can consume the same flat feature vectors used by `AudioResNet`.

    Accepted input shapes:
    - [B, D] flat feature vectors
    - [B, ...] tensors that can be flattened to [B, D]
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int | None = None,
        hidden_dims: List[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if input_dim is None:
            raise ValueError(
                "FineTunableYAMNet now follows the AudioResNet feature-vector scheme; "
                "`input_dim` is required."
            )

        if hidden_dims is None:
            hidden_dims = [128, 256, 256, 128]
        if len(hidden_dims) == 0:
            raise ValueError("`hidden_dims` must contain at least one layer width.")

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.bn_input = nn.BatchNorm1d(hidden_dims[0])

        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.res_blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i + 1], dropout=dropout)
            )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

    def _to_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            raise ValueError("Expected input with batch dimension, e.g. [B, D].")

        if x.dim() > 2:
            x = x.flatten(start_dim=1)

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected flattened feature dimension {self.input_dim}, got {x.shape[1]}."
            )
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_feature_vector(x)

        x = self.input_layer(x)
        x = self.bn_input(x)
        x = F.relu(x)
        x = self.dropout(x)

        for block in self.res_blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.dropout(x)
        return self.classifier(x)

    def freeze_backbone(self) -> None:
        for p in self.input_layer.parameters():
            p.requires_grad = False
        for p in self.bn_input.parameters():
            p.requires_grad = False
        for p in self.res_blocks.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.input_layer.parameters():
            p.requires_grad = True
        for p in self.bn_input.parameters():
            p.requires_grad = True
        for p in self.res_blocks.parameters():
            p.requires_grad = True

    def unfreeze_last_n_blocks(self, n_blocks: int = 4) -> None:
        self.freeze_backbone()
        modules: List[nn.Module] = [self.input_layer, self.bn_input] + list(self.res_blocks)
        n_blocks = max(0, min(n_blocks, len(modules)))
        for module in modules[-n_blocks:]:
            for p in module.parameters():
                p.requires_grad = True

    def parameter_groups(
        self,
        backbone_lr: float = 1e-4,
        head_lr: float = 1e-3,
    ) -> list[dict]:
        backbone_params = [
            p
            for module in [self.input_layer, self.bn_input] + list(self.res_blocks)
            for p in module.parameters()
            if p.requires_grad
        ]
        head_params = [p for p in self.classifier.parameters() if p.requires_grad]

        groups: list[dict] = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        if head_params:
            groups.append({"params": head_params, "lr": head_lr})
        return groups
