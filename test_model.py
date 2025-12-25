"""
VisiHealth Model Testing Script
Simple script to test your trained model on any medical image
"""

import torch
import yaml
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer
import json
import os
from tkinter import Tk, filedialog

# Import your models
from models import get_cnn_model, get_bert_model, build_visihealth_model
from utils.knowledge_graph import load_knowledge_graph, RationaleGenerator

def select_image():
    """Open file dialog to select an image"""
    
    print("\n📂 Opening file browser...")
    print("   Please select a medical image file")
    
    # Create Tkinter root window (hidden)
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select Medical Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()  # Close Tkinter
    
    if file_path:
        print(f"✅ Selected: {os.path.basename(file_path)}")
        return file_path
    else:
        print("❌ No file selected")
        return None


def load_model(checkpoint_path='checkpoints/best_checkpoint.pth', config_path='config.yaml'):
    """Load the trained model and configuration"""
    
    print("🔄 Loading model...")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get answer vocabulary from checkpoint or model info
    if 'answer_vocab' in checkpoint:
        answer_vocab = checkpoint['answer_vocab']
    else:
        # Try loading from VisiHealth_Model_Info.json
        info_path = 'results/VisiHealth_Model_Info.json'
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
                answer_vocab = {int(k): v for k, v in info['answer_vocab'].items()}
        else:
            raise ValueError("Answer vocabulary not found! Need either checkpoint with vocab or VisiHealth_Model_Info.json")
    
    num_classes = len(answer_vocab)
    
    # Update config with correct number of classes
    config['num_classes'] = num_classes
    config['model']['num_classes'] = num_classes
    config['model']['cnn']['num_classes'] = num_classes
    if 'fusion' not in config:
        config['fusion'] = {}
    config['fusion']['num_classes'] = num_classes
    
    # Build model
    cnn = get_cnn_model(config)
    bert = get_bert_model(config)
    model = build_visihealth_model(config, cnn, bert)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Device: {device}")
    print(f"   Classes: {num_classes}")
    print(f"   Trained epochs: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Best validation accuracy: {checkpoint.get('best_val_acc', 0):.2f}%")
    
    # Load Knowledge Graph for rationale generation
    kg = None
    rationale_gen = None
    kg_file = 'kg.txt'
    
    if os.path.exists(kg_file):
        print(f"\n🧠 Loading Knowledge Graph...")
        kg = load_knowledge_graph(kg_file)
        rationale_gen = RationaleGenerator(kg)
        print(f"   ✅ Loaded {len(kg.triplets)} knowledge triplets")
    else:
        print(f"\n⚠️  Knowledge Graph not found (kg.txt)")
        print(f"   Will provide answers without explanations")
    
    return model, answer_vocab, config, device, rationale_gen


def preprocess_image(image_path):
    """Preprocess image for the model"""
    
    # Image transformations (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def predict(model, image_path, question, answer_vocab, config, device, rationale_gen=None):
    """Make a prediction on an image and question"""
    
    print(f"\n{'='*60}")
    print(f"🔍 Processing your query...")
    print(f"{'='*60}")
    
    # Preprocess image
    image_tensor = preprocess_image(image_path).to(device)
    
    # Tokenize question
    tokenizer = AutoTokenizer.from_pretrained(config['model']['bert']['model_name'])
    encoded = tokenizer(
        question,
        padding='max_length',
        truncation=True,
        max_length=config['model']['bert']['max_length'],
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor, input_ids, attention_mask, return_attention=True)
        answer_logits = outputs['answer_logits']
        roi_scores = outputs['roi_scores']
    
    # Get prediction
    probs = torch.nn.functional.softmax(answer_logits, dim=1)
    pred_idx = answer_logits.argmax(dim=1).item()
    confidence = probs[0, pred_idx].item()
    
    # Get answer text
    predicted_answer = answer_vocab.get(pred_idx, f"Unknown (class {pred_idx})")
    
    # Get top 3 predictions
    top_k = torch.topk(probs[0], k=min(3, len(answer_vocab)))
    top_predictions = [
        (answer_vocab.get(idx.item(), f"Class {idx.item()}"), prob.item())
        for idx, prob in zip(top_k.indices, top_k.values)
    ]
    
    # Get top ROI
    top_roi_idx = roi_scores.argmax(dim=1).item()
    roi_confidence = torch.nn.functional.softmax(roi_scores, dim=1)[0, top_roi_idx].item()
    
    # Print results
    print(f"\n📸 Image: {os.path.basename(image_path)}")
    print(f"❓ Question: {question}")
    print(f"\n{'='*60}")
    print(f"💡 ANSWER: {predicted_answer}")
    print(f"📊 Confidence: {confidence*100:.2f}%")
    print(f"{'='*60}")
    
    print(f"\n🔝 Top 3 Predictions:")
    for i, (ans, prob) in enumerate(top_predictions, 1):
        print(f"   {i}. {ans} ({prob*100:.2f}%)")
    
    print(f"\n🎯 Top Region of Interest: ROI #{top_roi_idx} (confidence: {roi_confidence*100:.2f}%)")
    
    # Generate rationale if Knowledge Graph is available
    rationale = None
    if rationale_gen:
        print(f"\n{'='*60}")
        print(f"📝 EXPLANATION (using Knowledge Graph):")
        print(f"{'='*60}")
        
        # Get top 3 ROIs for rationale
        top_k_rois = torch.topk(roi_scores[0], k=min(3, roi_scores.shape[1]))
        
        rationale = rationale_gen.generate_rationale(
            predicted_answer=predicted_answer,
            confidence=confidence,
            top_roi_indices=top_k_rois.indices.tolist(),
            roi_scores=top_k_rois.values.tolist(),
            question=question
        )
        
        print(f"\n{rationale}")
        print(f"\n{'='*60}")
    
    print()
    
    return {
        'answer': predicted_answer,
        'confidence': confidence,
        'top_predictions': top_predictions,
        'top_roi': top_roi_idx,
        'roi_confidence': roi_confidence,
        'rationale': rationale
    }


def main():
    """Main function for interactive testing"""
    
    print("="*60)
    print("🏥 VisiHealth AI - Medical VQA Model Testing")
    print("="*60)
    
    # Load model
    model, answer_vocab, config, device, rationale_gen = load_model()
    
    print("\n" + "="*60)
    print("📋 How to use:")
    print("   1. Provide path to a medical image")
    print("   2. Ask a question about the image")
    print("   3. Get the AI's answer!")
    print("="*60)
    
    # Interactive loop
    while True:
        print("\n" + "-"*60)
        
        # Get image using file browser
        print("\n📁 Click to select a medical image...")
        image_path = select_image()
        
        if image_path is None:
            retry = input("\n🔄 No image selected. Try again? (yes/no): ").strip().lower()
            if retry not in ['yes', 'y', '']:
                print("\n👋 Goodbye!")
                break
            continue
        
        # Check if file exists (should always exist since we used file dialog)
        if not os.path.exists(image_path):
            print(f"❌ Error: File not found: {image_path}")
            continue
        
        # Get question
        question = input("❓ Enter your question: ").strip()
        
        if not question:
            print("❌ Error: Question cannot be empty!")
            continue
        
        # Make prediction
        try:
            result = predict(model, image_path, question, answer_vocab, config, device, rationale_gen)
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Ask if user wants to continue
        continue_test = input("\n🔄 Test another image? (yes/no): ").strip().lower()
        if continue_test not in ['yes', 'y', '']:
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
