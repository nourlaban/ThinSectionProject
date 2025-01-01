import os
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from datetime import datetime



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        self.encoder1 = conv_block(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = conv_block(512, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)
        
        self.output_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))
        
        b = self.bottleneck(self.pool4(e4))
        
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)
        
        return self.output_conv(d1)
    
import torch
import torch.nn as nn
import torchvision.models as models

class ResNetUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ResNetUNet, self).__init__()
        
        # Load pre-trained ResNet50 model
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Replace the first convolutional layer to handle n_channels
        self.first_conv = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Copy weights for shared channels if applicable
        if n_channels != 3:
            with torch.no_grad():
                self.first_conv.weight[:, :3] = self.resnet.conv1.weight
                if n_channels > 3:
                    self.first_conv.weight[:, 3:] = 0
        
        # Encoder (modified ResNet50 layers)
        self.encoder1 = nn.Sequential(
            self.first_conv,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4
        
        # Decoder
        self.decoder5 = self._make_decoder_block(2048, 1024)
        self.decoder4 = self._make_decoder_block(1024 + 1024, 512)
        self.decoder3 = self._make_decoder_block(512 + 512, 256)
        self.decoder2 = self._make_decoder_block(256 + 256, 128)
        self.decoder1 = self._make_decoder_block(128 + 64, 64)
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)

       

        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        # Decoder
        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat([d5, e4], dim=1))
        d3 = self.decoder3(torch.cat([d4, e3], dim=1))
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))

        # Pad e1 to match d2's spatial dimensions
        diffY = d2.size()[2] - e1.size()[2]
        diffX = d2.size()[3] - e1.size()[3]
        e1 = F.pad(e1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        d1 = self.decoder1(torch.cat([d2, e1], dim=1))
        
        # Adjust the size of e1 to match d2
        # d1 = self.decoder1(torch.cat([d2, self._center_crop(e1, d2.size()[2:])], dim=1))
        
        # Final output
        return self.final_conv(d1)
    
    def _center_crop(self, layer, target_size):
        _, _, h, w = layer.size()
        th, tw = target_size
        i = (h - th) // 2
        j = (w - tw) // 2
        return layer[:, :, i:i+th, j:j+tw]

class ResNetUNet_withBatch(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ResNetUNet_withBatch, self).__init__()
        
        # Load pre-trained ResNet50 model
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Replace the first convolutional layer to handle n_channels
        self.first_conv = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Copy weights for shared channels if applicable
        if n_channels != 3:
            with torch.no_grad():
                self.first_conv.weight[:, :3] = self.resnet.conv1.weight
                if n_channels > 3:
                    self.first_conv.weight[:, 3:] = 0
        
        # Encoder (modified ResNet50 layers)
        self.encoder1 = nn.Sequential(
            self.first_conv,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4
        
        # Decoder
        self.decoder5 = self._make_decoder_block(2048, 1024)
        self.decoder4 = self._make_decoder_block(1024 + 1024, 512)
        self.decoder3 = self._make_decoder_block(512 + 512, 256)
        self.decoder2 = self._make_decoder_block(256 + 256, 128)
        self.decoder1 = self._make_decoder_block(128 + 64, 64)
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        # Decoder
        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat([d5, e4], dim=1))
        d3 = self.decoder3(torch.cat([d4, e3], dim=1))
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))

        # Pad e1 to match d2's spatial dimensions
        diffY = d2.size()[2] - e1.size()[2]
        diffX = d2.size()[3] - e1.size()[3]
        e1 = F.pad(e1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        d1 = self.decoder1(torch.cat([d2, e1], dim=1))
        
        # Final output
        return self.final_conv(d1)
    
    def _center_crop(self, layer, target_size):
        _, _, h, w = layer.size()
        th, tw = target_size
        i = (h - th) // 2
        j = (w - tw) // 2
        return layer[:, :, i:i+th, j:j+tw]



class ResNextUNet_withBatch(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ResNextUNet_withBatch, self).__init__()
        
        # Load pre-trained ResNeXt101_64x4d model
        self.resnext = models.resnext101_64x4d(weights=models.ResNeXt101_64X4D_Weights.DEFAULT)
        
        # Replace the first convolutional layer to handle n_channels
        self.first_conv = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Copy weights for shared channels if applicable
        if n_channels != 3:
            with torch.no_grad():
                self.first_conv.weight[:, :3] = self.resnext.conv1.weight
                if n_channels > 3:
                    self.first_conv.weight[:, 3:] = 0
        
        # Encoder (modified ResNeXt101 layers)
        self.encoder1 = nn.Sequential(
            self.first_conv,
            self.resnext.bn1,
            self.resnext.relu,
            self.resnext.maxpool
        )
        self.encoder2 = self.resnext.layer1
        self.encoder3 = self.resnext.layer2
        self.encoder4 = self.resnext.layer3
        self.encoder5 = self.resnext.layer4
        
        # Decoder
        self.decoder5 = self._make_decoder_block(2048, 1024)
        self.decoder4 = self._make_decoder_block(1024 + 1024, 512)
        self.decoder3 = self._make_decoder_block(512 + 512, 256)
        self.decoder2 = self._make_decoder_block(256 + 256, 128)
        self.decoder1 = self._make_decoder_block(128 + 64, 64)
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        # Decoder
        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat([d5, e4], dim=1))
        d3 = self.decoder3(torch.cat([d4, e3], dim=1))
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))

        # Pad e1 to match d2's spatial dimensions
        diffY = d2.size()[2] - e1.size()[2]
        diffX = d2.size()[3] - e1.size()[3]
        e1 = F.pad(e1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        d1 = self.decoder1(torch.cat([d2, e1], dim=1))
        
        # Final output
        return self.final_conv(d1)
    
    def _center_crop(self, layer, target_size):
        _, _, h, w = layer.size()
        th, tw = target_size
        i = (h - th) // 2
        j = (w - tw) // 2
        return layer[:, :, i:i+th, j:j+tw]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNext101UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ResNext101UNet, self).__init__()
        
        # Load pre-trained ResNeXt101_64x4d model
        self.resnext = models.resnext101_64x4d(weights=models.ResNeXt101_64X4D_Weights.DEFAULT)
        
        # Replace the first convolutional layer to handle n_channels
        self.first_conv = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if n_channels != 3:
            with torch.no_grad():
                self.first_conv.weight[:, :3] = self.resnext.conv1.weight
                if n_channels > 3:
                    self.first_conv.weight[:, 3:] = 0
        
        # Encoder
        self.encoder1 = nn.Sequential(
            self.first_conv,
            self.resnext.bn1,
            self.resnext.relu,
            self.resnext.maxpool
        )
        self.encoder2 = self.resnext.layer1
        self.encoder3 = self.resnext.layer2
        self.encoder4 = self.resnext.layer3
        self.encoder5 = self.resnext.layer4
        
        # Decoder
        self.decoder5 = self._make_decoder_block(2048, 1024)
        self.decoder4 = self._make_decoder_block(1024 + 1024, 512)
        self.decoder3 = self._make_decoder_block(512 + 512, 256)
        self.decoder2 = self._make_decoder_block(256 + 256, 128)
        self.decoder1 = self._make_decoder_block(128 + 64, 64)
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        # Decoder
        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat([d5, e4], dim=1))
        d3 = self.decoder3(torch.cat([d4, e3], dim=1))
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))

        # Interpolation for matching dimensions
        e1 = F.interpolate(e1, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.decoder1(torch.cat([d2, e1], dim=1))
        
        # Final output
        return self.final_conv(d1)


class ResNetUNet_withBatch_Attention(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ResNetUNet_withBatch_Attention, self).__init__()
        
        # Load pre-trained ResNet50 model
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Replace the first convolutional layer to handle n_channels
        self.first_conv = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Copy weights for shared channels if applicable
        if n_channels != 3:
            with torch.no_grad():
                self.first_conv.weight[:, :3] = self.resnet.conv1.weight
                if n_channels > 3:
                    self.first_conv.weight[:, 3:] = 0
        
        # Encoder (modified ResNet50 layers)
        self.encoder1 = nn.Sequential(
            self.first_conv,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4
        
        # Decoder
        self.decoder5 = self._make_decoder_block(2048, 1024)
        self.decoder4 = self._make_decoder_block(1024 + 1024, 512)
        self.decoder3 = self._make_decoder_block(512 + 512, 256)
        self.decoder2 = self._make_decoder_block(256 + 256, 128)
        self.decoder1 = self._make_decoder_block(128 + 64, 64)
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            self._make_attention_block(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_attention_block(self, channels):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        print(f'e1: {e1.shape}')  # Debug
        e2 = self.encoder2(e1)
        print(f'e2: {e2.shape}')  # Debug
        e3 = self.encoder3(e2)
        print(f'e3: {e3.shape}')  # Debug
        e4 = self.encoder4(e3)
        print(f'e4: {e4.shape}')  # Debug
        e5 = self.encoder5(e4)
        print(f'e5: {e5.shape}')  # Debug
        
        # Decoder
        d5 = self.decoder5(e5)
        print(f'd5: {d5.shape}')  # Debug
        
        # Ensure e4 matches d5 spatial size
        e4_cropped = self._center_crop(e4, d5.shape[2:])
        print(f'e4_cropped: {e4_cropped.shape}')  # Debug
        d4 = self.decoder4(torch.cat([d5, e4_cropped], dim=1))
        print(f'd4: {d4.shape}')  # Debug

        e3_cropped = self._center_crop(e3, d4.shape[2:])
        print(f'e3_cropped: {e3_cropped.shape}')  # Debug
        d3 = self.decoder3(torch.cat([d4, e3_cropped], dim=1))
        print(f'd3: {d3.shape}')  # Debug

        e2_cropped = self._center_crop(e2, d3.shape[2:])
        print(f'e2_cropped: {e2_cropped.shape}')  # Debug
        d2 = self.decoder2(torch.cat([d3, e2_cropped], dim=1))
        print(f'd2: {d2.shape}')  # Debug

        e1_padded = self._pad_to_match(e1, d2)
        print(f'e1_padded: {e1_padded.shape}')  # Debug
        d1 = self.decoder1(torch.cat([d2, e1_padded], dim=1))
        print(f'd1: {d1.shape}')  # Debug

        # Final output
        out = self.final_conv(d1)
        print(f'out: {out.shape}')  # Debug
        
        # Ensure the output has the desired spatial dimensions
        # Optionally, adjust using interpolation if necessary
        # out = F.interpolate(out, size=(128, 128), mode='bilinear', align_corners=True)
        
        return out
    
    def _center_crop(self, layer, target_size):
        _, _, h, w = layer.size()
        th, tw = target_size
        i = (h - th) // 2
        j = (w - tw) // 2
        return layer[:, :, i:i+th, j:j+tw]
    
    def _pad_to_match(self, e1, d2):
        diffY = d2.size()[2] - e1.size()[2]
        diffX = d2.size()[3] - e1.size()[3]
        return F.pad(e1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

class ResNetUNet34_withBatch(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ResNetUNet34_withBatch, self).__init__()
        
        # Load pre-trained ResNet34 model
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # Replace the first convolutional layer to handle n_channels
        self.first_conv = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Copy weights for shared channels if applicable
        if n_channels != 3:
            with torch.no_grad():
                self.first_conv.weight[:, :3] = self.resnet.conv1.weight
                if n_channels > 3:
                    self.first_conv.weight[:, 3:] = 0
        
        # Encoder (modified ResNet34 layers)
        self.encoder1 = nn.Sequential(
            self.first_conv,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4
        
        # Decoder
        self.decoder5 = self._make_decoder_block(512, 256)
        self.decoder4 = self._make_decoder_block(256 + 256, 128)
        self.decoder3 = self._make_decoder_block(128 + 128, 64)
        self.decoder2 = self._make_decoder_block(64 + 64, 64)
        self.decoder1 = self._make_decoder_block(64 + 64, 64)
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        # Decoder
        d5 = self.decoder5(e5)
        d4 = self.decoder4(torch.cat([d5, e4], dim=1))
        d3 = self.decoder3(torch.cat([d4, e3], dim=1))
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))

        # Pad e1 to match d2's spatial dimensions
        diffY = d2.size()[2] - e1.size()[2]
        diffX = d2.size()[3] - e1.size()[3]
        e1 = F.pad(e1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        d1 = self.decoder1(torch.cat([d2, e1], dim=1))
        
        # Final output
        return self.final_conv(d1)
    
    def _center_crop(self, layer, target_size):
        _, _, h, w = layer.size()
        th, tw = target_size
        i = (h - th) // 2
        j = (w - tw) // 2
        return layer[:, :, i:i+th, j:j+tw]

class ResNetUNet_withBatch_dropout(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ResNetUNet_withBatch_dropout, self).__init__()
        
        # Load pre-trained ResNet50 model
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Replace the first convolutional layer to handle n_channels
        self.first_conv = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Copy weights for shared channels if applicable
        if n_channels != 3:
            with torch.no_grad():
                self.first_conv.weight[:, :3] = self.resnet.conv1.weight
                if n_channels > 3:
                    self.first_conv.weight[:, 3:] = 0
        
        # Encoder (modified ResNet50 layers)
        self.encoder1 = nn.Sequential(
            self.first_conv,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4
        
        # Decoder with attention
        self.decoder5 = self._make_decoder_block(2048, 1024)
        self.attention4 = self._make_attention_block(1024)
        self.decoder4 = self._make_decoder_block(1024 + 1024, 512)
        self.attention3 = self._make_attention_block(512)
        self.decoder3 = self._make_decoder_block(512 + 512, 256)
        self.attention2 = self._make_attention_block(256)
        self.decoder2 = self._make_decoder_block(256 + 256, 128)
        self.attention1 = self._make_attention_block(128)
        self.decoder1 = self._make_decoder_block(128 + 64, 64)
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_attention_block(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        # Decoder with attention
        d5 = self.decoder5(e5)
        a4 = self.attention4(d5)
        d4 = self.decoder4(torch.cat([d5 * a4, e4], dim=1))
        a3 = self.attention3(d4)
        d3 = self.decoder3(torch.cat([d4 * a3, e3], dim=1))
        a2 = self.attention2(d3)
        d2 = self.decoder2(torch.cat([d3 * a2, e2], dim=1))
        
        # Pad e1 to match d2's spatial dimensions
        diffY = d2.size()[2] - e1.size()[2]
        diffX = d2.size()[3] - e1.size()[3]
        e1 = F.pad(e1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        a1 = self.attention1(d2)
        d1 = self.decoder1(torch.cat([d2 * a1, e1], dim=1))
        
        # Final output
        return self.final_conv(d1)


class MulticlassHyperspectralDataset(Dataset):
    def __init__(self, image_dir, mask_dir, num_classes, transform=None, augmentation=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augmentation = augmentation
        self.num_classes = num_classes
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)
            image = np.moveaxis(image, 0, -1)
            image = image / 255.0  # Normalize

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.uint8)  # Use int64 for class labels

        # Ensure mask values are within valid range
        unique_values = np.unique(mask)
        if np.max(unique_values) >= self.num_classes or np.min(unique_values) < 0:
            print(f"Warning: Invalid class labels in mask file {self.mask_files[idx]}")
            mask = np.clip(mask, 0, self.num_classes - 1)

        image = to_tensor(image)
        mask = torch.from_numpy(mask).long()

        if self.augmentation:
            # Apply augmentations separately
            image = self.augmentation(image)
            mask = self.augmentation(mask.unsqueeze(0)).squeeze(0)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask.unsqueeze(0)).squeeze(0)

        return image, mask
    
def calculate_metrics(outputs, targets, num_classes):
    # Convert outputs to class predictions
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    #targets = torch.argmax(targets, dim=1).cpu().numpy()
    
    # Flatten the arrays
    preds = preds.ravel()
    targets = targets.ravel()
    
    # Calculate metrics
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, average='weighted', zero_division=0)
    recall = recall_score(targets, preds, average='weighted', zero_division=0)
    f1 = f1_score(targets, preds, average='weighted', zero_division=0)
    
    return accuracy, precision, recall, f1

def train_model(model, train_loader, val_loader, device, num_classes, epochs=10):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_accuracy = 0.0
        train_precision = 0.0
        train_recall = 0.0
        train_f1 = 0.0
        num_batches = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Calculate metrics
            accuracy, precision, recall, f1 = calculate_metrics(outputs, masks, num_classes)
            train_accuracy += accuracy
            train_precision += precision
            train_recall += recall
            train_f1 += f1
            num_batches += 1
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, "
              f"Accuracy: {train_accuracy/num_batches}, "
              f"Precision: {train_precision/num_batches}, "
              f"Recall: {train_recall/num_batches}, "
              f"F1: {train_f1/num_batches}")

        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_precision = 0.0
        val_recall = 0.0
        val_f1 = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calculate metrics
                accuracy, precision, recall, f1 = calculate_metrics(outputs, masks, num_classes)
                val_accuracy += accuracy
                val_precision += precision
                val_recall += recall
                val_f1 += f1
                num_val_batches += 1
        
        print(f"Validation Loss: {val_loss/len(val_loader)}, "
              f"Validation Accuracy: {val_accuracy/num_val_batches}, "
              f"Validation Precision: {val_precision/num_val_batches}, "
              f"Validation Recall: {val_recall/num_val_batches}, "
              f"Validation F1: {val_f1/num_val_batches}")
import matplotlib.pyplot as plt

def train_model_with_graph(model, train_loader, val_loader, device, num_classes, epochs=10):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate accuracy
            preds = torch.argmax(outputs, dim=1)
            running_accuracy += (preds == masks).float().mean().item()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = running_accuracy / len(train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Calculate accuracy
                preds = torch.argmax(outputs, dim=1)
                val_accuracy += (preds == masks).float().mean().item()
        
        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, "
              f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    torch.save(model.state_dict(), f'trained_model_{timestamp}.pth')

    # Plotting loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plotting accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, epochs+1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Save the plot
    plt.savefig(f'metrics_{timestamp}.png')

    # plt.show()

# Example usage remains the same

def train_model_lr(model, train_loader, val_loader,result_dir, device, num_classes, epochs=10, learning_rate=1e-3):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Example: Using a StepLR scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Initialize metric lists
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        running_precision = 0.0
        running_recall = 0.0
        num_batches = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate metrics
            accuracy, precision, recall, _ = calculate_metrics(outputs, masks, num_classes)
            running_accuracy += accuracy
            running_precision += precision
            running_recall += recall
            num_batches += 1

        scheduler.step()  # Update learning rate

        # Aggregate metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = running_accuracy / num_batches
        train_precision = running_precision / num_batches
        train_recall = running_recall / num_batches
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_precision = 0.0
        val_recall = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Calculate metrics
                accuracy, precision, recall, _ = calculate_metrics(outputs, masks, num_classes)
                val_accuracy += accuracy
                val_precision += precision
                val_recall += recall
                num_val_batches += 1

        # Aggregate validation metrics
        val_loss /= len(val_loader)
        val_accuracy /= num_val_batches
        val_precision /= num_val_batches
        val_recall /= num_val_batches
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    # Save the trained model
    torch.save(model.state_dict(), os.path.join(result_dir,f'trained_model_{timestamp}.pth'))
    # Plotting loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plotting accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, epochs+1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Save the plot
    # Generate a timestamp
  

    # Save the plot with the timestamp
    plt.savefig( os.path.join(result_dir,f'metrics_{timestamp}.png'))
    # plt.show()


# Example usage with a different learning rate

def predict_and_save(model, root_image_dir, output_dir, device, input_shape, num_classes):
    os.makedirs(output_dir, exist_ok=True)
    image_dir =  os.path.join(root_image_dir,"all_tiles/images")
    image_files = sorted(os.listdir(image_dir))
    
    model.to(device)
    model.eval()

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        with rasterio.open(image_path) as src:
            img = src.read().astype(np.float32)
            img = np.moveaxis(img, 0, -1)
            img = img / 255.0  # Normalize
            
            img_tensor = ToTensor()(img)
            img_resized = Resize(input_shape)(img_tensor)
            
            img_input = img_resized.unsqueeze(0).to(device)

            with torch.no_grad():
                prediction = model(img_input)
                prediction = F.softmax(prediction, dim=1)
                prediction = prediction.squeeze().cpu().numpy()
                prediction = np.argmax(prediction, axis=0).astype(np.uint8)

                meta = src.meta.copy()
                meta.update({
                    'count': 1,
                    'dtype': 'uint8',
                    'height': prediction.shape[0],
                    'width': prediction.shape[1]
                })

                output_path = os.path.join(output_dir, f"pred_{image_file}")
                with rasterio.open(output_path, 'w', **meta) as dst:
                    dst.write(prediction, 1)


if __name__ == "__main__":
    # Example usage
    datadir = r'F:\Senaa\thensections\6bands\output_tiles'
    num_channels = 6  # Adjust based on your hyperspectral data
    num_classes = 12  # Adjust based on your number of classes
    input_shape = (64, 64)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #unet_model = UNet(n_channels=num_channels, n_classes=num_classes)

    # Usage
    #unet_model = ResNetUNet_withBatch(n_channels=num_channels, n_classes=num_classes)  # Now works with any number of input channels
    unet_model = ResNetUNet(n_channels=num_channels, n_classes=num_classes)  # Now works with any number of input channels



    # Define data augmentation pipeline
    augmentation_pipeline = Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(degrees=5)
    ])

    # Usage with data augmentation
    train_dataset = MulticlassHyperspectralDataset(
        os.path.join(datadir, 'train', 'images'),
        os.path.join(datadir, 'train', 'masks'),
        num_classes=num_classes,
        transform=Resize(input_shape),
        augmentation=augmentation_pipeline
    )

    val_dataset = MulticlassHyperspectralDataset(
        os.path.join(datadir, 'val', 'images'),
        os.path.join(datadir, 'val', 'masks'),
        num_classes=num_classes,
        transform=Resize(input_shape)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    ##train_model_lr(unet_model, train_loader, val_loader, device, num_classes, epochs=200)
    #train_model_lr(unet_model, train_loader, val_loader, device, num_classes, epochs=50, learning_rate=5e-4)





    # Load the state dictionary
    unet_model.load_state_dict(torch.load('trained_model_20241024_150338.pth'))
    # Predict and save
    test_image_dir = os.path.join(datadir, 'all_tiles', 'images')
    output_dir = os.path.join(datadir, 'all_tiles', 'predictions')
    predict_and_save(unet_model, test_image_dir, output_dir, device, input_shape, num_classes)