"""
VisiHealth AI - SLAKE Dataset Preprocessing
Handles loading, augmentation, and tokenization for the SLAKE dataset
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from transformers import AutoTokenizer
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional


class SLAKEDataset(Dataset):
    """
    SLAKE Dataset Loader with Augmentation
    
    Handles:
    - Loading 642 radiology images
    - Processing 14,028 bilingual QA pairs (English subset)
    - Applying data augmentation to mitigate overfitting
    - Tokenizing questions with BioBERT
    - Loading segmentation masks for 39 organs/12 diseases
    - Accessing 2,600+ KG triplets
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        tokenizer_name: str = 'michiyasunaga/BioLinkBERT-base',
        max_length: int = 128,
        image_size: int = 224,
        augment: bool = True,
        language: str = 'en',
        answer_vocab: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            data_dir: Root directory containing SLAKE dataset
            split: 'train', 'validate', or 'test'
            tokenizer_name: Pretrained BioBERT tokenizer
            max_length: Maximum question length
            image_size: Image resize dimension
            augment: Whether to apply data augmentation
            language: 'en' for English, 'ch' for Chinese
            answer_vocab: Pre-built answer vocabulary (use training vocab for val/test)
        """
        super(SLAKEDataset, self).__init__()
        
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        self.language = language
        self.max_length = max_length
        
        # Paths
        self.images_dir = os.path.join(data_dir, 'imgs')
        self.masks_dir = os.path.join(data_dir, 'masks')
        
        # Load QA pairs
        qa_file = os.path.join(data_dir, f'{split}.json')
        print(f"Loading {split} data from {qa_file}")
        with open(qa_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Filter by language if bilingual dataset
        self.data = [item for item in self.data if item.get('q_lang', 'en') == language]
        
        print(f"Loaded {len(self.data)} QA pairs for {split} split")
        
        # Initialize BioBERT tokenizer
        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Build or use provided answer vocabulary
        if answer_vocab is None:
            # Build vocabulary from current split
            self.answer_vocab = self._build_answer_vocab()
        else:
            # Use provided vocabulary (for val/test to match train)
            self.answer_vocab = answer_vocab
        
        self.num_classes = len(self.answer_vocab)
        print(f"Answer vocabulary size: {self.num_classes}")
        
        # Define image transforms
        self.train_transform = self._get_train_transforms()
        self.val_transform = self._get_val_transforms()
        
        # Load segmentation mask info (if available)
        self.has_masks = os.path.exists(self.masks_dir)
        
    def _build_answer_vocab(self) -> Dict[str, int]:
        """
        Build vocabulary from all answers in training data
        Maps answer strings to class indices
        """
        answers = set()
        
        # Collect all unique answers
        for item in self.data:
            answer = str(item['answer']).lower().strip()
            answers.add(answer)
        
        # Create mapping
        answer_vocab = {ans: idx for idx, ans in enumerate(sorted(answers))}
        
        return answer_vocab
    
    def _get_train_transforms(self):
        """
        Training augmentation pipeline to mitigate overfitting
        Includes rotation, flipping, color jittering, and normalization
        """
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            
            # Data Augmentation (critical for small dataset)
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            
            # Convert to tensor and normalize
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            ),
            
            # Random erasing for regularization
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
        ])
    
    def _get_val_transforms(self):
        """Validation/Test transforms without augmentation"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_image(self, img_name: str) -> Image.Image:
        """Load image from disk"""
        img_path = os.path.join(self.images_dir, img_name)
        
        # Handle different file extensions
        if not os.path.exists(img_path):
            # Try with .jpg extension
            img_path = os.path.join(self.images_dir, f"{img_name}.jpg")
        if not os.path.exists(img_path):
            # Try with .png extension
            img_path = os.path.join(self.images_dir, f"{img_name}.png")
            
        image = Image.open(img_path).convert('RGB')
        return image
    
    def _load_mask(self, img_name: str) -> Optional[np.ndarray]:
        """Load segmentation mask if available"""
        if not self.has_masks:
            return None
        
        mask_path = os.path.join(self.masks_dir, f"{img_name}_mask.png")
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize((self.image_size, self.image_size))
            return np.array(mask)
        return None
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        
        Returns:
            Dictionary containing:
                - image: Augmented image tensor [3, 224, 224]
                - question: Tokenized question (input_ids, attention_mask)
                - answer: Answer class index
                - mask: Segmentation mask (if available)
                - metadata: Original question text, answer text, image name
        """
        item = self.data[idx]
        
        # Extract fields
        img_name = item.get('img_name', item.get('img_id', ''))
        question_text = item['question']
        answer_text = str(item['answer']).lower().strip()
        
        # Load and transform image
        image = self._load_image(img_name)
        if self.augment:
            image_tensor = self.train_transform(image)
        else:
            image_tensor = self.val_transform(image)
        
        # Tokenize question using BioBERT tokenizer
        question_tokens = self.tokenizer(
            question_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get answer class
        answer_idx = self.answer_vocab.get(answer_text, 0)  # Default to 0 if unknown
        
        # Load segmentation mask (for multi-task learning)
        mask = self._load_mask(img_name)
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).float() / 255.0
        else:
            # Create dummy mask if not available
            mask_tensor = torch.zeros((self.image_size, self.image_size))
        
        # Prepare sample
        sample = {
            'image': image_tensor,
            'input_ids': question_tokens['input_ids'].squeeze(0),
            'attention_mask': question_tokens['attention_mask'].squeeze(0),
            'answer': torch.tensor(answer_idx, dtype=torch.long),
            'mask': mask_tensor,
            'question_text': question_text,
            'answer_text': answer_text,
            'img_name': img_name,
            'question_type': item.get('answer_type', 'CLOSED'),
            'content_type': item.get('content_type', 'unknown')
        }
        
        return sample
    
    def get_answer_text(self, answer_idx: int) -> str:
        """Convert answer index back to text"""
        idx_to_answer = {v: k for k, v in self.answer_vocab.items()}
        return idx_to_answer.get(answer_idx, 'unknown')


def get_dataloader(
    data_dir: str,
    split: str,
    batch_size: int = 16,
    num_workers: int = 4,
    tokenizer_name: str = 'michiyasunaga/BioLinkBERT-base',
    train_vocab: Optional[Dict[str, int]] = None,
    **kwargs
) -> Tuple[DataLoader, SLAKEDataset]:
    """
    Create dataloader for SLAKE dataset
    
    Args:
        data_dir: Path to SLAKE dataset
        split: 'train', 'validate', or 'test'
        batch_size: Batch size
        num_workers: Number of data loading workers
        tokenizer_name: BioBERT tokenizer name
        train_vocab: Training vocabulary for val/test splits (ensures consistency)
    
    Returns:
        DataLoader and Dataset objects
    """
    # Create dataset
    dataset = SLAKEDataset(
        data_dir=data_dir,
        split=split,
        tokenizer_name=tokenizer_name,
        augment=(split == 'train'),
        answer_vocab=train_vocab,
        **kwargs
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader, dataset


if __name__ == "__main__":
    # Test the dataset
    print("Testing SLAKEDataset...")
    
    # Example usage
    data_dir = "./data/SLAKE"
    
    try:
        # Create dataset
        train_loader, train_dataset = get_dataloader(
            data_dir=data_dir,
            split='train',
            batch_size=4,
            num_workers=0  # Use 0 for testing
        )
        
        print(f"\nDataset created successfully!")
        print(f"Number of samples: {len(train_dataset)}")
        print(f"Number of batches: {len(train_loader)}")
        print(f"Number of answer classes: {train_dataset.num_classes}")
        
        # Get first batch
        batch = next(iter(train_loader))
        
        print(f"\nBatch structure:")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Attention mask shape: {batch['attention_mask'].shape}")
        print(f"  Answer shape: {batch['answer'].shape}")
        print(f"  Mask shape: {batch['mask'].shape}")
        
        print(f"\nSample question: {batch['question_text'][0]}")
        print(f"Sample answer: {batch['answer_text'][0]}")
        print(f"Sample image: {batch['img_name'][0]}")
        
    except FileNotFoundError as e:
        print(f"\nDataset files not found: {e}")
        print("Please download the SLAKE dataset and place it in ./data/SLAKE/")
        print("Expected structure:")
        print("  ./data/SLAKE/")
        print("    ├── imgs/")
        print("    ├── masks/")
        print("    ├── train.json")
        print("    ├── validate.json")
        print("    └── test.json")
