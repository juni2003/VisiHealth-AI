"""
VisiHealth AI - Knowledge Graph System
Handles KG triplet indexing, retrieval, and rationale generation
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
from collections import defaultdict


class KnowledgeGraph:
    """
    Medical Knowledge Graph Handler
    
    Manages 2,600+ KG triplets from SLAKE:
    - Index triplets by entities and relations
    - Retrieve relevant facts based on ROI and question
    - Score triplets by relevance
    """
    
    def __init__(self, kg_file: str):
        """
        Args:
            kg_file: Path to knowledge graph file (.txt or .json)
        """
        self.kg_file = kg_file
        self.triplets = []
        self.entity_index = defaultdict(list)  # entity -> list of triplet indices
        self.relation_index = defaultdict(list)  # relation -> list of triplet indices
        
        self._load_knowledge_graph()
        self._build_indices()
        
        print(f"Loaded {len(self.triplets)} KG triplets")
    
    def _load_knowledge_graph(self):
        """Load KG triplets from file"""
        if not os.path.exists(self.kg_file):
            print(f"Warning: KG file not found at {self.kg_file}")
            return
        
        with open(self.kg_file, 'r', encoding='utf-8') as f:
            if self.kg_file.endswith('.json'):
                data = json.load(f)
                # Assume format: [{"head": "...", "relation": "...", "tail": "..."}, ...]
                for item in data:
                    self.triplets.append({
                        'head': item['head'],
                        'relation': item['relation'],
                        'tail': item['tail']
                    })
            else:
                # Assume tab-separated format: head\trelation\ttail
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        self.triplets.append({
                            'head': parts[0].strip(),
                            'relation': parts[1].strip(),
                            'tail': parts[2].strip()
                        })
    
    def _build_indices(self):
        """Build entity and relation indices for fast retrieval"""
        for idx, triplet in enumerate(self.triplets):
            head = triplet['head'].lower()
            tail = triplet['tail'].lower()
            relation = triplet['relation'].lower()
            
            self.entity_index[head].append(idx)
            self.entity_index[tail].append(idx)
            self.relation_index[relation].append(idx)
    
    def retrieve_by_entity(self, entity: str, topk: int = 3) -> List[Dict]:
        """
        Retrieve triplets containing a specific entity
        
        Args:
            entity: Entity name (e.g., "liver", "lung")
            topk: Number of triplets to return
        
        Returns:
            List of relevant triplets
        """
        entity = entity.lower()
        indices = self.entity_index.get(entity, [])
        
        triplets = [self.triplets[idx] for idx in indices[:topk]]
        return triplets
    
    def retrieve_by_keywords(self, keywords: List[str], topk: int = 3) -> List[Dict]:
        """
        Retrieve triplets matching any keyword
        
        Args:
            keywords: List of keywords to search
            topk: Number of triplets to return
        
        Returns:
            Ranked list of relevant triplets
        """
        scores = defaultdict(float)
        
        for keyword in keywords:
            keyword = keyword.lower()
            
            # Search in entities
            if keyword in self.entity_index:
                for idx in self.entity_index[keyword]:
                    scores[idx] += 1.0
            
            # Partial matching
            for entity, indices in self.entity_index.items():
                if keyword in entity:
                    for idx in indices:
                        scores[idx] += 0.5
        
        # Sort by score
        sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top-k triplets
        triplets = [self.triplets[idx] for idx, score in sorted_indices[:topk]]
        return triplets
    
    def retrieve_for_question_and_roi(
        self,
        question: str,
        roi_names: List[str],
        topk: int = 3
    ) -> List[Dict]:
        """
        Retrieve relevant KG triplets based on question and detected ROIs
        
        Args:
            question: Question text
            roi_names: List of detected ROI names (e.g., ["liver", "kidney"])
            topk: Number of triplets to return
        
        Returns:
            List of most relevant KG triplets
        """
        # Extract keywords from question
        question_keywords = self._extract_keywords(question)
        
        # Combine ROI and question keywords
        all_keywords = list(set(roi_names + question_keywords))
        
        # Retrieve triplets
        triplets = self.retrieve_by_keywords(all_keywords, topk=topk * 2)
        
        # Re-rank by relevance to both question and ROI
        scored_triplets = []
        for triplet in triplets:
            score = self._score_triplet_relevance(triplet, question_keywords, roi_names)
            scored_triplets.append((triplet, score))
        
        # Sort and return top-k
        scored_triplets.sort(key=lambda x: x[1], reverse=True)
        return [triplet for triplet, score in scored_triplets[:topk]]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Simple keyword extraction (can be improved with NLP)
        stop_words = {'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'this', 'that', 'there', 'what', 'which',
                     'who', 'where', 'when', 'how', 'can', 'does'}
        
        words = text.lower().split()
        keywords = [w.strip('?.,!') for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def _score_triplet_relevance(
        self,
        triplet: Dict,
        question_keywords: List[str],
        roi_names: List[str]
    ) -> float:
        """
        Score how relevant a triplet is to the question and ROI
        
        Higher score = more relevant
        """
        score = 0.0
        
        head = triplet['head'].lower()
        tail = triplet['tail'].lower()
        relation = triplet['relation'].lower()
        
        # Check ROI matches (higher weight)
        for roi in roi_names:
            roi = roi.lower()
            if roi in head or roi in tail:
                score += 2.0
        
        # Check question keyword matches
        for keyword in question_keywords:
            if keyword in head or keyword in tail or keyword in relation:
                score += 1.0
        
        return score


class RationaleGenerator:
    """
    Generates human-readable rationales from predictions and KG facts
    """
    
    # ROI ID to name mapping (39 organs from SLAKE)
    ROI_NAMES = {
        0: 'liver', 1: 'kidney', 2: 'spleen', 3: 'pancreas', 4: 'gallbladder',
        5: 'stomach', 6: 'intestine', 7: 'lung', 8: 'heart', 9: 'aorta',
        10: 'brain', 11: 'skull', 12: 'spine', 13: 'muscle', 14: 'bone',
        15: 'fat', 16: 'vessel', 17: 'nerve', 18: 'lymph', 19: 'tumor',
        20: 'cyst', 21: 'lesion', 22: 'nodule', 23: 'mass', 24: 'fluid',
        25: 'air', 26: 'calcification', 27: 'hemorrhage', 28: 'infarct',
        29: 'edema', 30: 'inflammation', 31: 'infection', 32: 'fracture',
        33: 'dislocation', 34: 'degeneration', 35: 'atrophy', 36: 'hypertrophy',
        37: 'stenosis', 38: 'occlusion'
    }
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        """
        Args:
            knowledge_graph: Initialized KnowledgeGraph object
        """
        self.kg = knowledge_graph
    
    def generate_rationale(
        self,
        predicted_answer: str,
        confidence: float,
        top_roi_indices: List[int],
        roi_scores: List[float],
        question: str,
        attention_map: Optional[torch.Tensor] = None
    ) -> str:
        """
        Generate human-friendly rationale
        
        Args:
            predicted_answer: Model's predicted answer
            confidence: Prediction confidence score
            top_roi_indices: Indices of top detected ROIs
            roi_scores: Confidence scores for each ROI
            question: Original question text
            attention_map: Visual attention map (optional)
        
        Returns:
            Human-readable rationale string
        """
        # Get ROI names
        roi_names = [self.ROI_NAMES.get(idx, f'region_{idx}') 
                     for idx in top_roi_indices[:3]]
        
        # Retrieve relevant KG triplets
        kg_triplets = self.kg.retrieve_for_question_and_roi(
            question=question,
            roi_names=roi_names,
            topk=1
        )
        
        # Build rationale
        rationale_parts = []
        
        # Part 1: Detected ROI
        if roi_scores[0] > 0.3:  # Threshold for confident detection
            main_roi = roi_names[0]
            rationale_parts.append(f"Detected {main_roi} region")
            if roi_scores[0] > 0.7:
                rationale_parts[-1] += f" with high confidence ({roi_scores[0]:.2f})"
        
        # Part 2: KG fact
        if kg_triplets:
            triplet = kg_triplets[0]
            kg_fact = f"{triplet['head']} {triplet['relation']} {triplet['tail']}"
            rationale_parts.append(f"Medical knowledge indicates: {kg_fact}")
        
        # Part 3: Conclusion
        confidence_text = "high" if confidence > 0.8 else "moderate" if confidence > 0.5 else "low"
        rationale_parts.append(f"Therefore, the answer is '{predicted_answer}' (confidence: {confidence_text})")
        
        # Combine parts
        rationale = ". ".join(rationale_parts) + "."
        
        return rationale
    
    def generate_batch_rationales(
        self,
        predictions: Dict,
        questions: List[str],
        answer_vocab: Dict[int, str]
    ) -> List[str]:
        """
        Generate rationales for a batch of predictions
        
        Args:
            predictions: Model prediction dictionary
            questions: List of question texts
            answer_vocab: Mapping from answer index to answer text
        
        Returns:
            List of rationale strings
        """
        rationales = []
        
        batch_size = len(questions)
        predicted_answers = predictions['predicted_answers']
        answer_probs = predictions['answer_probs']
        top_rois = predictions['top_rois']
        roi_scores = predictions['roi_scores']
        
        for i in range(batch_size):
            # Get answer text
            answer_idx = predicted_answers[i].item()
            answer_text = answer_vocab.get(answer_idx, 'unknown')
            
            # Get confidence
            confidence = answer_probs[i, answer_idx].item()
            
            # Generate rationale
            rationale = self.generate_rationale(
                predicted_answer=answer_text,
                confidence=confidence,
                top_roi_indices=top_rois[i].tolist(),
                roi_scores=roi_scores[i].tolist(),
                question=questions[i]
            )
            
            rationales.append(rationale)
        
        return rationales


def load_knowledge_graph(kg_file: str) -> KnowledgeGraph:
    """Factory function to load knowledge graph"""
    return KnowledgeGraph(kg_file)


if __name__ == "__main__":
    # Test the knowledge graph system
    print("Testing Knowledge Graph System...")
    
    # Example KG file (create sample if not exists)
    kg_file = "./data/SLAKE/knowledge_graph.txt"
    
    # Create sample KG if file doesn't exist
    if not os.path.exists(kg_file):
        os.makedirs(os.path.dirname(kg_file), exist_ok=True)
        with open(kg_file, 'w', encoding='utf-8') as f:
            f.write("liver\tis_located_in\tabdomen\n")
            f.write("hepatomegaly\taffects\tliver\n")
            f.write("enlarged_liver\tindicates\thepatomen\n")
            f.write("lung\tis_located_in\tchest\n")
            f.write("pneumonia\taffects\tlung\n")
            f.write("kidney\tis_located_in\tabdomen\n")
        print(f"Created sample KG file at {kg_file}")
    
    # Load KG
    kg = load_knowledge_graph(kg_file)
    
    # Test retrieval
    print("\n--- Test Entity Retrieval ---")
    triplets = kg.retrieve_by_entity("liver", topk=2)
    for t in triplets:
        print(f"  {t['head']} -> {t['relation']} -> {t['tail']}")
    
    print("\n--- Test Question + ROI Retrieval ---")
    question = "Is there an enlarged liver in this CT scan?"
    roi_names = ["liver", "abdomen"]
    triplets = kg.retrieve_for_question_and_roi(question, roi_names, topk=2)
    for t in triplets:
        print(f"  {t['head']} -> {t['relation']} -> {t['tail']}")
    
    # Test rationale generation
    print("\n--- Test Rationale Generation ---")
    rationale_gen = RationaleGenerator(kg)
    
    rationale = rationale_gen.generate_rationale(
        predicted_answer="Yes",
        confidence=0.85,
        top_roi_indices=[0, 1, 2],  # liver, kidney, spleen
        roi_scores=[0.9, 0.6, 0.3],
        question="Is there an enlarged liver?"
    )
    
    print(f"Generated Rationale:\n  {rationale}")
