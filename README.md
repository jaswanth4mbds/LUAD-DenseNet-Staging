# **LUAD-DenseNet-Staging**

A deep-learning pipeline for **automated staging of Lung Adenocarcinoma (LUAD)** using **whole-slide histopathology images**.  
A **DenseNet121** model pretrained on **ImageNet** and fine-tuned on **TCGA-LUAD** patches that classifies **Normal and Stage IA–IV**.  
It includes code for **patch extraction, stain normalization, training, and inference**.

---

### 📥 **Download Model Weights**
🔗 Model weights (.pth): https://drive.google.com/file/d/14DN2UD91y9nOOI_8of7fD-YWKbRHo2qZ/view?usp=drive_link

---

### **Project Structure**
src/  
 ├── training.py  
 ├── patches.py  
 └── normalization.py  

models/  
 └── best_model.pth   (download link provided above)  

requirements.txt

---

### ✅ **How to Use**

**1. Install Dependencies**  
    pip install -r requirements.txt

**2. Extract Patches (optional)**  
    python src/patches.py --input_folder <WSI_folder> --output_folder patches

**3. Stain Normalize (optional)**  
    python src/normalization.py

**4. Train Model**  
    python src/training.py --data_dir data --epochs 30 --batch_size 32 --lr 1e-4

---

### 🧠 **Model Information**
- **Model:** DenseNet121  
- **Dataset:** TCGA-LUAD  
- **Task:** 8-class LUAD staging  
- **Classes:** Normal, IA, IB, IIA, IIB, IIIA, IIIB, IV

---

### 📊 **Data Source**
TCGA-LUAD whole-slide histology dataset  
(WSI files are not included. They can be accessed from **TCGA**.)

---

### ⚠️ **Notes**
- For **academic and research use only**
- **Not for clinical use** without proper validation

