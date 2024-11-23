# AI-Assisted Pulmonary Fibrosis Detection  

## **Overview**  
This project aims to assist radiologists in detecting **Idiopathic Pulmonary Fibrosis (IPF)** through **Artificial Intelligence (AI)-powered solutions**. By analyzing CT scan images, our tool identifies subtle patterns of IPF that may be challenging to detect with the naked eye. The solution not only enhances diagnostic accuracy but also accelerates the process, improving patient outcomes.

---

## **Features**  
- **Advanced Image Analysis:** AI models capable of identifying early signs of IPF.  
- **Automated Report Generation:** Detailed diagnostic reports to support radiologists' clinical judgment.  
- **High Sensitivity to Subtle Patterns:** Detection of hard-to-spot fibrosis indicators.  
- **User-Friendly Interface:** Easy integration into radiologists' workflows.  

---

Machine Learning Part:
  - Kaggle Data + Clinical Data Sources
      ↓
  - Advanced Threshold-Based Segmentation (Dynamic FVC Clustering)
      ↓
  - Train CV Model (with Explainability Features)
      ↓
  - ML Regression for Patient Data Integration
      ↓
  - Merge Outputs for Final Probability Scoring
      ↓
  - LLM for Clinical Report Generation

App Interface:
  - Input: CT Scans + Patient Data
      ↓
  - Processing: Probability Calculation + LLM Summary
      ↓
  - Output: 
      - Final Probability (e.g., "82%")
      - LLM-Generated Report
      - Visual Highlights on CT Scans

---

