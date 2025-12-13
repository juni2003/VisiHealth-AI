# VisiHealth AI  
**Making Medical AI Human-Friendly**

VisiHealth AI is a Final Year Project (FYP) that proposes a **knowledge-enhanced Medical Visual Question Answering (Med-VQA) system** aimed at improving explainability and trust in AI-assisted healthcare. The system is designed to answer clinical questions based on medical images while providing **human-readable rationales grounded in medical knowledge and visual evidence**.

---

## Project Overview

Traditional Med-VQA systems focus primarily on prediction accuracy and often function as black-box models, offering little to no explanation for their decisions. This lack of transparency limits their adoption in real-world clinical environments where interpretability and evidence-based reasoning are critical.

VisiHealth AI addresses this limitation by integrating:
- Medical image understanding
- Natural language question processing
- Knowledge graph–based reasoning
- Region-of-Interest (ROI) localization
- Template-based rationale generation

The goal is to produce not only accurate answers but also **verifiable explanations** that can be trusted by medical professionals and students.

---

## Dataset

The project is based on the **SLAKE 1.0 (English subset)** dataset, which uniquely supports explainability-focused research. It provides:
- Medical radiology images (X-ray, CT, MRI, Ultrasound)
- Question–Answer (QA) pairs
- Semantic segmentation masks for organs and diseases
- Medical knowledge graph triplets

This rich annotation enables multimodal learning and grounded reasoning.

---

## Proposed Methodology (Conceptual)

The proposed system follows a multimodal pipeline:
1. **Image Encoder**  
   A CNN trained from scratch to extract visual features and localize clinically relevant regions using segmentation masks.

2. **Text Encoder**  
   A fine-tuned BioBERT/ClinicalBERT model to encode clinical questions.

3. **Knowledge Graph Retrieval**  
   Relevant medical facts are retrieved using similarity-based ranking from KG triplets.

4. **Fusion Module**  
   Visual, textual, and knowledge embeddings are combined to predict the final answer.

5. **Rationale Generation**  
   A template-based engine generates human-readable explanations grounded in ROI and KG facts.

---

## Project Objectives

- Develop a multimodal Med-VQA architecture combining vision, language, and knowledge.
- Improve explainability using ROI localization and knowledge-based reasoning.
- Enhance trust and interpretability in medical AI systems.
- Provide an interactive web-based prototype for demonstration purposes.

---

## Repository Status

📌 **Current Stage:**  
Initial project proposal uploaded.  
Implementation, experiments, and evaluations will be added incrementally.

---

## Folder Structure (Planned)

- `documentation/` — Proposal, reports, and academic documents  
- `data/` — Dataset references and preprocessing outputs  
- `models/` — Vision, text, and fusion models  
- `src/` — Core implementation code  
- `web/` — Web application (frontend + backend)  
- `results/` — Evaluation outputs and visualizations  

---

## Team Members

- **Junaid Mohi Ud Din** — 01-134222-071  
- **Hammad ur Rehman** — 01-134222-059  


---

## Supervisor

**Sir Abdul Rahman**  
Department of Computer Science  
Bahria University, Islamabad



---

## Disclaimer

This project is developed for academic and research purposes only and is not intended for clinical deployment or real-world diagnosis.
