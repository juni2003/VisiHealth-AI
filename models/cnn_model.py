"""
VisiHealth AI - Custom CNN Model
Trains from scratch for medical image feature extraction and ROI localization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic Convolutional Block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ROIAttention(nn.Module):
    """Region of Interest Attention Module"""
    def __init__(self, in_channels):
        super(ROIAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attention_map = self.attention(x)
        attended_features = x * attention_map
        return attended_features, attention_map


class MedicalCNN(nn.Module):
    """
    Custom CNN for Medical Image Analysis
    - Extracts global image features
    - Localizes Regions of Interest (ROI)
    - Supports multi-task learning with segmentation
    """
    def __init__(self, config):
        super(MedicalCNN, self).__init__()
        
        self.config = config
        dropout = config.get('dropout', 0.5)
        feature_dim = config.get('feature_dim', 512)
        
        # Encoder: Feature Extraction
        self.layer1 = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2)  # 112x112
        )
        
        self.layer2 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2, 2)  # 56x56
        )
        
        self.layer3 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2, 2)  # 28x28
        )
        
        self.layer4 = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            nn.MaxPool2d(2, 2)  # 14x14
        )
        
        self.layer5 = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            nn.MaxPool2d(2, 2)  # 7x7
        )
        
        # ROI Attention Modules
        self.roi_attention_layer3 = ROIAttention(256)
        self.roi_attention_layer4 = ROIAttention(512)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature Projection
        self.feature_projection = nn.Sequential(
            nn.Linear(512 + 256 + 512, feature_dim),  # Concat global + ROI features
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Segmentation Head (for multi-task learning)
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),  # Binary segmentation
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass
        Args:
            x: Input image tensor [B, 3, 224, 224]
            return_attention: Whether to return attention maps
        Returns:
            features: Fused feature vector [B, feature_dim]
            segmentation_mask: Predicted segmentation mask (if multitask)
            attention_maps: ROI attention maps (if return_attention=True)
        """
        # Encoder forward pass
        x1 = self.layer1(x)  # [B, 64, 112, 112]
        x2 = self.layer2(x1)  # [B, 128, 56, 56]
        x3 = self.layer3(x2)  # [B, 256, 28, 28]
        x4 = self.layer4(x3)  # [B, 512, 14, 14]
        x5 = self.layer5(x4)  # [B, 512, 7, 7]
        
        # Global features
        global_features = self.global_pool(x5).squeeze(-1).squeeze(-1)  # [B, 512]
        
        # ROI Attention
        roi_features_3, attention_map_3 = self.roi_attention_layer3(x3)
        roi_features_3 = self.global_pool(roi_features_3).squeeze(-1).squeeze(-1)  # [B, 256]
        
        roi_features_4, attention_map_4 = self.roi_attention_layer4(x4)
        roi_features_4 = self.global_pool(roi_features_4).squeeze(-1).squeeze(-1)  # [B, 512]
        
        # Concatenate global and ROI features
        combined_features = torch.cat([global_features, roi_features_3, roi_features_4], dim=1)
        
        # Project to final feature dimension
        features = self.feature_projection(combined_features)  # [B, feature_dim]
        
        # Segmentation mask (for multi-task learning)
        segmentation_mask = self.segmentation_head(x5)
        
        outputs = {
            'features': features,
            'segmentation_mask': segmentation_mask
        }
        
        if return_attention:
            outputs['attention_maps'] = {
                'layer3': attention_map_3,
                'layer4': attention_map_4
            }
        
        return outputs


class ROILocalizer(nn.Module):
    """
    Extracts and localizes specific Regions of Interest
    Used for grounding rationales in specific image regions
    """
    def __init__(self, feature_dim=512, num_rois=39):  # 39 organs in SLAKE
        super(ROILocalizer, self).__init__()
        
        self.roi_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_rois),
            nn.Sigmoid()  # Multi-label classification
        )
        
    def forward(self, features):
        """
        Args:
            features: CNN features [B, feature_dim]
        Returns:
            roi_scores: Probability scores for each ROI [B, num_rois]
        """
        return self.roi_classifier(features)


def get_cnn_model(config):
    """Factory function to create CNN model"""
    return MedicalCNN(config['model']['cnn'])


if __name__ == "__main__":
    # Test the model
    config = {
        'model': {
            'cnn': {
                'dropout': 0.5,
                'feature_dim': 512
            }
        }
    }
    
    model = get_cnn_model(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    outputs = model(dummy_input, return_attention=True)
    
    print(f"Features shape: {outputs['features'].shape}")
    print(f"Segmentation mask shape: {outputs['segmentation_mask'].shape}")
    print(f"Attention maps: {list(outputs['attention_maps'].keys())}")
