# **LUAD-DenseNet-Staging**

A deep-learning pipeline for **automated staging of Lung Adenocarcinoma (LUAD)** using **whole-slide histopathology images**.  
A **DenseNet121** model pretrained on **ImageNet** and fine-tuned on **TCGA-LUAD** patches that classifies **Normal and Stage IA–IV**.  
It includes code for **patch extraction, stain normalization, training, and inference**.

---

### **Download Model Weights**
🔗 Model weights (.pth): https://drive.google.com/file/d/14DN2UD91y9nOOI_8of7fD-YWKbRHo2qZ/view?usp=drive_link

The trained model achieved 89.6% accuracy and a macro F1-score of 0.889 on the test set.

---

### **Project Structure**
src/  
 ├── training.py  
 ├── patches.py  
 └── normalization.py  
 
---

### **How to Use**

**1. Install Dependencies**  
    pip install -r requirements.txt

**2. Extract Patches**  
    python src/patches.py --input_folder <WSI_folder> --output_folder patches

**3. Stain Normalize**  
    python src/normalization.py

**4. Train Model**  
    python src/training.py --data_dir data --epochs 30 --batch_size 32 --lr 1e-4

---

### **Model Information**
- **Model:** DenseNet121  
- **Dataset:** TCGA-LUAD  
- **Task:** 8-class LUAD staging  
- **Classes:** Normal, IA, IB, IIA, IIB, IIIA, IIIB, IV

---

### **Data Source**
TCGA-LUAD whole-slide histology dataset  
(WSI files are not included. They can be accessed from **TCGA**.)

This work uses slide-level TCGA clinical stage labels without pixel-level or region-level annotations. Patches are weakly labeled based on their parent WSI following a standard weak-supervision approach for histopathology.

---

### **Notes**
- For **academic and research use only**
- **Not for clinical use** without proper validation

