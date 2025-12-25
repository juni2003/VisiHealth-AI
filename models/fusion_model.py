"""
VisiHealth AI - Multimodal Fusion and Answer Prediction
Combines CNN visual features and BERT question embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFusion(nn.Module):
    """
    Fuses visual and textual features for answer prediction
    Supports multiple fusion strategies
    """
    def __init__(self, visual_dim, text_dim, fusion_method='concat', hidden_dims=[512, 256]):
        super(MultimodalFusion, self).__init__()
        
        self.fusion_method = fusion_method
        
        if fusion_method == 'concat':
            # Simple concatenation
            input_dim = visual_dim + text_dim
            
        elif fusion_method == 'bilinear':
            # Bilinear pooling
            self.bilinear = nn.Bilinear(visual_dim, text_dim, hidden_dims[0])
            input_dim = hidden_dims[0]
            
        elif fusion_method == 'attention':
            # Attention-based fusion
            self.visual_attention = nn.Sequential(
                nn.Linear(visual_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 1)
            )
            self.text_attention = nn.Sequential(
                nn.Linear(text_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 1)
            )
            input_dim = visual_dim + text_dim
        
        # Fusion layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
            ])
            prev_dim = hidden_dim
        
        self.fusion_layers = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        
    def forward(self, visual_features, text_features):
        """
        Fuse visual and textual features
        Args:
            visual_features: CNN features [B, visual_dim]
            text_features: BERT embeddings [B, text_dim]
        Returns:
            fused_features: Combined features [B, output_dim]
        """
        if self.fusion_method == 'concat':
            fused = torch.cat([visual_features, text_features], dim=1)
            
        elif self.fusion_method == 'bilinear':
            fused = self.bilinear(visual_features, text_features)
            
        elif self.fusion_method == 'attention':
            # Compute attention weights
            v_att = torch.softmax(self.visual_attention(visual_features), dim=0)
            t_att = torch.softmax(self.text_attention(text_features), dim=0)
            
            # Apply attention
            v_weighted = visual_features * v_att
            t_weighted = text_features * t_att
            
            fused = torch.cat([v_weighted, t_weighted], dim=1)
        
        # Pass through fusion layers
        fused_features = self.fusion_layers(fused)
        
        return fused_features


class AnswerPredictor(nn.Module):
    """
    Predicts answers from fused multimodal features
    Supports both classification (closed-ended) and generation (open-ended)
    """
    def __init__(self, input_dim, num_classes, answer_type='classification'):
        super(AnswerPredictor, self).__init__()
        
        self.answer_type = answer_type
        
        if answer_type == 'classification':
            # For closed-ended questions (Yes/No, specific categories)
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:
            # For open-ended questions (can be extended to generation)
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, fused_features):
        """
        Predict answer
        Args:
            fused_features: Fused multimodal features [B, input_dim]
        Returns:
            logits: Answer prediction logits [B, num_classes]
        """
        logits = self.classifier(fused_features)
        return logits


class VisiHealthModel(nn.Module):
    """
    Complete VisiHealth AI Model
    Integrates CNN, BERT, Fusion, and Answer Prediction
    """
    def __init__(self, cnn_model, bert_model, config):
        super(VisiHealthModel, self).__init__()
        
        self.cnn = cnn_model
        self.bert = bert_model
        
        # Get dimensions
        visual_dim = config['model']['cnn']['feature_dim']
        text_dim = config['model']['bert']['hidden_size']
        
        # Fusion
        fusion_config = config['model']['fusion']
        self.fusion = MultimodalFusion(
            visual_dim=visual_dim,
            text_dim=text_dim,
            fusion_method=fusion_config['method'],
            hidden_dims=fusion_config['hidden_dims']
        )
        
        # Answer Prediction
        # Note: num_classes should be determined from the dataset vocabulary
        self.answer_predictor = AnswerPredictor(
            input_dim=self.fusion.output_dim,
            num_classes=config['model']['cnn'].get('num_classes', 1000)
        )
        
        # ROI Localizer (for grounding rationales)
        from models.cnn_model import ROILocalizer
        self.roi_localizer = ROILocalizer(visual_dim, num_rois=39)
        
    def forward(self, images, input_ids, attention_mask, return_attention=False):
        """
        Complete forward pass
        Args:
            images: Input images [B, 3, 224, 224]
            input_ids: Tokenized questions [B, max_length]
            attention_mask: Attention masks [B, max_length]
            return_attention: Whether to return attention maps
        Returns:
            Dictionary containing predictions and intermediate outputs
        """
        # Extract visual features
        cnn_outputs = self.cnn(images, return_attention=return_attention)
        visual_features = cnn_outputs['features']
        
        # Extract textual features
        text_features = self.bert(input_ids, attention_mask)
        
        # Fuse features
        fused_features = self.fusion(visual_features, text_features)
        
        # Predict answer
        answer_logits = self.answer_predictor(fused_features)
        
        # Localize ROI
        roi_scores = self.roi_localizer(visual_features)
        
        outputs = {
            'answer_logits': answer_logits,
            'roi_scores': roi_scores,
            'visual_features': visual_features,
            'text_features': text_features,
            'fused_features': fused_features,
            'segmentation_mask': cnn_outputs['segmentation_mask']
        }
        
        if return_attention:
            outputs['attention_maps'] = cnn_outputs.get('attention_maps', None)
        
        return outputs
    
    def predict(self, images, questions, device='cuda'):
        """
        High-level prediction interface
        Args:
            images: Input images (PIL or tensor)
            questions: List of question strings
            device: Device to run on
        Returns:
            Predictions dictionary
        """
        self.eval()
        with torch.no_grad():
            # Tokenize questions
            tokens = self.bert.tokenize(questions)
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            
            if not isinstance(images, torch.Tensor):
                # Convert PIL images to tensor if needed
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
                images = torch.stack([transform(img) for img in images])
            
            images = images.to(device)
            
            # Forward pass
            outputs = self.forward(images, input_ids, attention_mask, return_attention=True)
            
            # Get predictions
            answer_probs = torch.softmax(outputs['answer_logits'], dim=1)
            predicted_answers = torch.argmax(answer_probs, dim=1)
            
            # Get top ROIs
            roi_probs = outputs['roi_scores']
            top_rois = torch.topk(roi_probs, k=3, dim=1)
            
            return {
                'predicted_answers': predicted_answers.cpu(),
                'answer_probs': answer_probs.cpu(),
                'top_rois': top_rois.indices.cpu(),
                'roi_scores': top_rois.values.cpu(),
                'attention_maps': outputs.get('attention_maps'),
                'segmentation_mask': outputs['segmentation_mask'].cpu()
            }


def build_visihealth_model(config, cnn_model, bert_model):
    """Factory function to build complete model"""
    return VisiHealthModel(cnn_model, bert_model, config)


if __name__ == "__main__":
    # Test the complete model
    import yaml
    from models.cnn_model import get_cnn_model
    from models.bert_model import get_bert_model
    
    # Load config
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create models
    cnn = get_cnn_model(config)
    bert = get_bert_model(config)
    model = build_visihealth_model(config, cnn, bert)
    
    print(f"Complete model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    questions = ["Is there an abnormality?", "What organ is affected?"]
    
    tokens = bert.tokenize(questions)
    outputs = model(images, tokens['input_ids'], tokens['attention_mask'])
    
    print(f"Answer logits shape: {outputs['answer_logits'].shape}")
    print(f"ROI scores shape: {outputs['roi_scores'].shape}")
