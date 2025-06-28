# 🍅 Xception for Tomato Leaf Disease Detection
**Enhanced Deep Learning Model with Bayesian Optimization**  
*IEEE ICAIQSA 2024 Conference Paper*

> 📄 [Read the Paper (IEEE)](https://doi.org/10.1109/ICAIQSA64000.2024.10882346)  
> 🔗 [Dataset (Kaggle)](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf) | 🧠 [Plant Village Repository](https://plantvillage.psu.edu/)  
> 📊 **97.23% Accuracy** | 🏆 **Best Performance on 10-Class Classification**

---

## 📌 Abstract

Tomato cultivation is essential to world agriculture, affecting food security and economic stability. Accurate detection of tomato leaf diseases is crucial to maintain crop health and maximize yield. This work improves disease classification by using deep learning models, notably CNN, Xception, DenseNet, and a **hyper-tuned Xception model**. 

We employed transfer learning and data augmentation to improve model performance on the Kaggle dataset, which contains **10,000 training images and 1,000 validation images** from the Plant Village Repository organized into **ten disease classes**. The hyper-tuned Xception model achieved the maximum performance by meticulously tweaking hyperparameters such as learning rate, batch size, and network architecture, with an **accuracy of 97.23%**, precision of 95.73%, recall of 94.40%, and F1-score of 95.06%.

These results show a considerable improvement over standard CNN, Xception, and DenseNet models, highlighting the efficiency of sophisticated deep learning algorithms and hyperparameter optimization in achieving high classification accuracy.

---

## ✨ Highlights
- 📊 Achieved **97.23% accuracy** on 10-class tomato leaf dataset
- ⚡ Hybrid optimization using **Bayesian Optimization + PBT**
- 🧠 Architecture enhancements with **Dilated Convolutions + SE Blocks**
- 🧪 Compared against CNN, DenseNet, and baseline Xception
- 🔬 Open-sourced notebooks for reproducibility and extension
---

## 📚 Introduction

**Motivation:**  
Tomato leaf diseases significantly impact agricultural productivity and food security globally. Early and accurate detection is crucial for timely intervention and crop protection.

**Why Xception?**  
Xception's depthwise separable convolutions provide an optimal balance between computational efficiency and high accuracy, making it ideal for real-world agricultural applications where resources may be limited.

**Research Gap:**  
While existing models show promise, there's a need for enhanced architectures that combine advanced optimization techniques with efficient neural networks to achieve superior performance in multi-class plant disease classification.

**Our Contribution:**  
We propose a **hybrid optimization approach** combining Bayesian Optimization with Population-Based Training (PBT) to fine-tune Xception, achieving state-of-the-art results on tomato leaf disease detection.

---

## 🛠️ Methodology

### **Model Architecture**
- **Base Model:** Xception (pre-trained on ImageNet)
- ![image](https://github.com/user-attachments/assets/288a488e-b620-492f-a4c8-36635a41334e)
- **Enhancement:** Modified architecture with dilated convolutions and Squeeze-and-Excitation (SE) blocks
- **Optimization:** Bayesian Optimization + Population-Based Training (PBT)
- **Training Strategy:** Transfer learning with fine-tuning

### **Key Technical Innovations**

#### 1. **Depthwise Separable Convolutions**
Xception replaces standard convolutions with depthwise separable convolutions, dramatically reducing parameters and computational costs:
- **Depthwise Convolution:** Each input channel applies its filter individually
- **Pointwise Convolution:** 1×1 kernel processes individual pixels
- ![image](https://github.com/user-attachments/assets/2cfdb3d7-1566-4c4a-93ec-103979af3e84)
- **Result:** Maintains accuracy while reducing computational overhead

#### 2. **Hybrid Hyperparameter Optimization**
Our novel approach combines:
- **Bayesian Optimization:** Global exploration of hyperparameter space
- **Population-Based Training:** Dynamic tuning during training
- **Benefits:** 2-3% accuracy improvement, 20-30% faster convergence

#### 3. **Enhanced Architecture Features**
- **Dilated Convolutions:** Increased receptive field with fewer parameters
- **SE Blocks:** Adaptive feature map recalibration
- **Custom Dense Layers:** Optimized regularization with batch normalization and dropout

### **Pipeline Overview**
```
Raw Images → Preprocessing → Data Augmentation → 
Modified Xception → Bayesian Optimization → 
Population-Based Training → Final Model → Evaluation
```

---

## 🧪 Dataset

**Source:** Kaggle Tomato Leaf Disease Detection (Plant Village Repository)

**Dataset Statistics:**
- **Training Set:** 10,000 images (1,000 per class)
- **Validation Set:** 1,000 images (100 per class)
- **Classes:** 10 disease categories
- **Format:** RGB images, various resolutions

### **Disease Classes:**
| Class | Description |
|-------|-------------|
| Healthy | Normal tomato leaves |
| Bacterial Spot | *Xanthomonas* infection |
| Early Blight | *Alternaria solani* |
| Late Blight | *Phytophthora infestans* |
| Septoria Leaf Spot | *Septoria lycopersici* |
| Tomato Mosaic Virus | Viral infection |
| Tomato Yellow Leaf Curl Virus | Viral infection |
| Target Spot | *Corynespora cassiicola* |
| Leaf Mold | *Passalora fulva* |
| Spider Mites | Two-spotted spider mite |

![image](https://github.com/user-attachments/assets/c91f45bc-1349-44d1-97ec-2595f65a03a6)


**Data Augmentation:**
- Rotation, zoom, horizontal flip
- Brightness and contrast adjustment
- Random cropping and resizing

---

## 📈 Results

### **Model Performance Comparison**

| Model | Epochs | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|-------|--------|--------------|---------------|------------|--------------|
| CNN | 27 | 86.44 | 84.82 | 83.97 | 84.39 |
| Standard Xception | 27 | 87.39 | 85.90 | 84.30 | 85.09 |
| DenseNet | 27 | 96.90 | 92.07 | 94.10 | 93.07 |
| **Our Method** | **27** | **97.23** | **95.73** | **94.40** | **95.06** |

![image](https://github.com/user-attachments/assets/29bf653e-9a1f-4ea3-b599-afd9e2310c27)


### **Optimized Hyperparameters**

| Parameter | Optimal Value |
|-----------|---------------|
| units_dense1 | 448 |
| dropout | 0.2 |
| units_dense2 | 160 |
| learning_rate | 0.000791 |

### **Key Achievements**
- 🏆 **97.23% Accuracy** - Best performance among compared models
- ⚡ **20-30% Faster Training** - Due to hybrid optimization
- 🎯 **Balanced Performance** - High precision and recall across all classes
- 💾 **Efficient Architecture** - Reduced parameters with maintained accuracy

---

## 🚀 How to Run

### **Installation**
```bash
git clone https://github.com/YashMaini30/Tomato_Leaf_Disease_Detection.git
cd Tomato_Leaf_Disease_Detection
pip install tensorflow scikit-learn opencv-python matplotlib seaborn numpy pandas scikit-optimize
```

### **Dataset Setup**
```bash
# Download dataset from Kaggle
kaggle datasets download -d kaustubhb999/tomatoleafdisease
unzip tomatoleafdisease.zip -d data/
```

### **Notebook Execution Order**

#### **1. Model Fine-tuning**
```bash
# Start with Xception fine-tuning
jupyter notebook Copy_of_models.ipynb
```
*This notebook contains the core Xception fine-tuning implementation*

#### **2. Bayesian Optimization & PBT**
```bash
# Run the main implementation with optimization
jupyter notebook Copy_of_Tomato_Leaf_detection.ipynb
```
*This notebook includes Bayesian optimization and Population-Based Training*

#### **3. Performance Comparison**
```bash
# Compare hypertuned model with baseline models
jupyter notebook Copy_of_final_implementation.ipynb
```
*This notebook shows the performance comparison achieving 97.23% accuracy*

### **Quick Start**
1. **Download the dataset** and place in `data/` folder
2. **Run notebooks in order** (models → main implementation → final comparison)
3. **Check results** in the final implementation notebook for all performance metrics

---

## 📊 Model Architecture Details

### **Modified Xception Enhancements**
1. **Dilated Convolutions:** Increased receptive field
2. **SE Blocks:** Channel attention mechanism
3. **Custom Dense Layers:** Optimized classification head
4. **Skip Connections:** Improved gradient flow

### **Mathematical Formulations**

**Dilated Convolution:**
```
y[i,j] = Σ x[i + r·m, j + r·n] · k[m,n]
```

**Squeeze-and-Excitation:**
```
s = F_ex(z, W) = σ(W2 · δ(W1 · z))
x̃ = F_scale(u_c, s_c) = s_c · u_c
```

---

## 📁 Project Structure

```
tomato-leaf-disease-detection/
├── Copy_of_Tomato_Leaf_detection.ipynb    # 🧠 Main implementation with Bayesian optimization and PBT
├── Copy_of_final_implementation.ipynb     # 📊 Performance comparison of hypertuned model vs others
├── Copy_of_models.ipynb                   # 🔧 Xception fine-tuning implementation
├── README.md                              # 📖 Project documentation
└── data/                                  # 📁 Dataset folder (download separately)
    ├── train/                             #     Training images by disease class
    └── validation/                        #     Validation images by disease class
```

---

## 🔬 Notebook Guide

### **📓 Copy_of_models.ipynb**
*Core Xception fine-tuning implementation*
- Base Xception architecture setup
- Transfer learning from ImageNet
- Model architecture modifications
- Initial fine-tuning experiments

### **🧠 Copy_of_Tomato_Leaf_detection.ipynb** 
*Main implementation with advanced optimization*
- Bayesian hyperparameter optimization
- Population-Based Training (PBT) implementation
- Enhanced Xception with dilated convolutions
- Squeeze-and-Excitation blocks integration
- Complete training pipeline

### **📊 Copy_of_final_implementation.ipynb**
*Performance comparison and final results*
- Comparison of CNN, Xception, DenseNet, and our method
- **97.23% accuracy achievement**
- Comprehensive evaluation metrics
- Results visualization and analysis
- Final model performance validation

### **🎯 Execution Workflow**
1. **Start with models.ipynb** → Understand base implementation
2. **Run Tomato_Leaf_detection.ipynb** → Train optimized model  
3. **Check final_implementation.ipynb** → See performance results

---

## 📖 Citation

If you use this work in your research, please cite:

```apa
Y. Maini, S. K. Singh and P. Saxena, "Xception for Tomato Leaf Disease Detection: Hyperparameter Tuning and Fine-tuning Approaches," 2024 International Conference on Artificial Intelligence and Quantum Computation-Based Sensor Application (ICAIQSA), Nagpur, India, 2024, pp. 1-6, doi: 10.1109/ICAIQSA64000.2024.10882346. keywords: {Deep learning;Training;Accuracy;Quantum computing;Biological system modeling;Computational modeling;Transfer learning;Stability analysis;Tuning;Diseases;Leaf Image Analysis;hyperparameter;deep learning;Transfer Learning;Plant Disease Diagnosis},
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## 📧 Contact

- **Email:** yash.13019011622@ipu.ac.in
- **Institution:** USAR GGSIPU
- **Conference:** IEEE ICAIQSA 2024

---

## 🙏 Acknowledgments

- **Dataset:** Plant Village Repository and Kaggle community
- **Framework:** TensorFlow/Keras team
- **Conference:** IEEE ICAIQSA 2024 organizers
- **Inspiration:** Agricultural AI research community
- **Motivation:** Dr. S.K. Singh and Dr. P. Saxena 

---

*Built with ❤️ for sustainable agriculture and food security*
