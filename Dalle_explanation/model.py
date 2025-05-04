import torch.nn as nn
import torch
import math    


class ClassificationNet1(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationNet1, self).__init__()
        
        self.conv1_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(512)
        self.conv1_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(512)
        self.conv1_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn1_3 = nn.BatchNorm2d(512)
        self.conv1_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn1_4 = nn.BatchNorm2d(512)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(512)
        self.conv2_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(512)
        self.conv2_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn2_3 = nn.BatchNorm2d(512)
        self.conv2_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn2_4 = nn.BatchNorm2d(512)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(512 * 4 * 4, 4096)
        self.bn_fc1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.bn_fc2 = nn.BatchNorm1d(4096)
        self.fc3 = nn.Linear(4096, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.relu(self.bn1_3(self.conv1_3(x)))
        x = self.relu(self.bn1_4(self.conv1_4(x)))
        x = self.maxpool1(x)
        
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.relu(self.bn2_3(self.conv2_3(x)))
        x = self.relu(self.bn2_4(self.conv2_4(x)))
        x = self.maxpool2(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)
        return x


    
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        out = x.mean(dim=[2, 3])  # Global Average Pooling
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out)).view(batch_size, num_channels, 1, 1)
        return x * out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.se = SEBlock(out_channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.se(out) 
        out = self.relu(out)
        return out

class ClassificationNet2(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationNet2, self).__init__()
        
        self.layer1 = self._make_layer(256, 512, 3)
        self.layer2 = self._make_layer(512, 512, 3)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 2048)  
        self.bn_fc1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.4)  
        
    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=2))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


import torch
from torch import nn
import math

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        mlp_output = self.mlp(x)
        x = x + self.dropout(mlp_output)
        x = self.norm2(x)

        return x

class ClassificationNet3(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, mlp_dim=2048, num_layers=6, dropout=0.1):
        super(ClassificationNet3, self).__init__()

        self.embed_dim = embed_dim
        self.num_patches = 16 * 16 

        self.proj = nn.Linear(256, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = self._build_sinusoidal_embeddings(self.num_patches + 1, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.pre_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.fc = nn.Linear(embed_dim * 2, num_classes)

    def _build_sinusoidal_embeddings(self, num_positions, embed_dim):
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(num_positions, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        # x shape: [batch_size, 256, 16, 16]
        batch_size = x.size(0)
        
        x = x.reshape(batch_size, 256, -1).transpose(1, 2)  # [batch_size, 256, embed_dim]
        
 
        x = self.proj(x)  

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :].to(x.device)

        x = self.dropout(x)

        for layer in self.encoder:
            x = layer(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]

        x = self.pre_classifier(cls_token_final)
 
        x = self.fc(x)

        return x



if __name__ == "__main__":
    model = ClassificationNet1(1000)
    print("model", model)
    input_tensor = torch.randn(32, 256, 16, 16)
    output = model(input_tensor)
    print(output.shape)