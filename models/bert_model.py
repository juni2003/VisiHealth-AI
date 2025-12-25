"""
VisiHealth AI - BERT Text Encoder
Fine-tunes BioBERT/ClinicalBERT for medical question encoding
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig


class MedicalBERTEncoder(nn.Module):
    """
    Medical Domain BERT Encoder
    Fine-tunes BioBERT or ClinicalBERT for question understanding
    """
    def __init__(self, config):
        super(MedicalBERTEncoder, self).__init__()
        
        self.config = config
        model_name = config.get('model_name', 'michiyasunaga/BioLinkBERT-base')
        hidden_size = config.get('hidden_size', 768)
        dropout = config.get('dropout', 0.3)
        freeze_layers = config.get('freeze_layers', 6)
        
        # Load pretrained BERT model
        print(f"Loading pretrained model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze early layers for efficiency
        if freeze_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for i in range(freeze_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        Args:
            input_ids: Tokenized question IDs [B, max_length]
            attention_mask: Attention mask [B, max_length]
        Returns:
            question_embeddings: Encoded question features [B, hidden_size]
        """
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # [B, hidden_size]
        
        # Project to final representation
        question_embeddings = self.projection(cls_output)
        
        return question_embeddings
    
    def tokenize(self, questions, max_length=128):
        """
        Tokenize questions
        Args:
            questions: List of question strings
            max_length: Maximum sequence length
        Returns:
            Dictionary with input_ids and attention_mask
        """
        return self.tokenizer(
            questions,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )


def get_bert_model(config):
    """Factory function to create BERT encoder"""
    return MedicalBERTEncoder(config['model']['bert'])


if __name__ == "__main__":
    # Test the model
    config = {
        'model': {
            'bert': {
                'model_name': 'michiyasunaga/BioLinkBERT-base',
                'hidden_size': 768,
                'dropout': 0.3,
                'freeze_layers': 6
            }
        }
    }
    
    model = get_bert_model(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Test tokenization and forward pass
    questions = [
        "Is there an enlarged liver in this CT scan?",
        "What organ is affected by the abnormality?"
    ]
    
    tokens = model.tokenize(questions)
    print(f"Input IDs shape: {tokens['input_ids'].shape}")
    
    embeddings = model(tokens['input_ids'], tokens['attention_mask'])
    print(f"Question embeddings shape: {embeddings.shape}")
