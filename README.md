# Nuclear Reaction Optimization (NRO) for Gene Selection

### Official Repository of the Method Published in *Diagnostics (2025)*

**"Evaluating the Nuclear Reaction Optimization (NRO) Algorithm for Gene Selection in Cancer Classification"**
Alkamli & Alshamlan, 2025

---

## 📌 Overview

This repository presents the methodology, datasets, and results of the **Nuclear Reaction Optimization (NRO)** algorithm applied to **gene selection for microarray cancer datasets**. NRO is a physics-inspired metaheuristic that simulates **nuclear fission** and **nuclear fusion** processes to explore and refine gene subsets.

This study was the first to evaluate NRO as a standalone gene selection method on microarray data, without prior dimensionality reduction. It forms the foundation of the subsequent hybrid methods:

- **F-NRO** — F-score filtering combined with NRO (*Current Issues in Molecular Biology*, 2025)
- **GNR** — Genetic-embedded NRO with F-score filtering (*International Journal of Molecular Sciences*, 2025)

---

## 🔒 Code Availability

The implementation of the NRO algorithm is available upon request for academic and research purposes.

To request access, please contact: shahad.s.alkamli@gmail.com

---

## 📁 Repository Structure

```
NRO/
│
├── NRO.py               # Code availability notice
│
├── Datasets/
│     ├── Colon.arff
│     ├── Leukemia1.arff
│     ├── Leukemia2.arff
│     ├── Lung.arff
│     ├── Lymphoma.arff
│     └── SRBCT.arff
│
└── README.md
```

---

## 🔬 Methodology

### **1. Preprocessing**

- Load `.arff` microarray datasets
- Handle missing values using mean imputation (Lymphoma dataset contains 4.91% missing values)
- Normalize features using Z-score
- Encode class labels numerically

### **2. Optimization Using NRO**

The algorithm alternates between two phases:

#### 🔹 Nuclear Fission (Exploration)

- Gaussian perturbation around candidate solutions
- Mutation factors for subaltern and essential fission products
- Step sizes that decrease progressively across generations

#### 🔹 Nuclear Fusion (Exploitation)

- Ionization based on differences between candidate solutions
- Fusion of promising solutions
- Lévy flight to escape local optima when solutions become too similar

### **3. Evaluation**

Each candidate gene subset is evaluated with:

- **Support Vector Machine (SVM)** with a linear kernel
- **k-Nearest Neighbors (k-NN)** with k = 5
- **Leave-One-Out Cross-Validation (LOOCV)**

The fitness function balances classification accuracy against the number of selected genes.

### **4. Repetition**

All experiments repeat **30 runs per dataset and classifier** for statistical reliability.

---

## 📊 Datasets

NRO is evaluated on six well-known microarray datasets:

| Dataset    | Classes | Samples | Genes |
|------------|---------|---------|-------|
| Colon      | 2       | 62      | 2000  |
| Leukemia 1 | 2       | 72      | 7129  |
| Leukemia 2 | 3       | 72      | 7129  |
| Lung       | 2       | 96      | 7129  |
| Lymphoma   | 3       | 62      | 4026  |
| SRBCT      | 4       | 83      | 2308  |

---

## 📈 Published Results (Diagnostics 2025)

Best classification accuracy achieved by NRO across 30 runs:

| Dataset    | SVM        | k-NN       |
|------------|------------|------------|
| Colon      | 82.16%     | 76.11%     |
| Leukemia 1 | 95.53%     | 78.84%     |
| Leukemia 2 | 92.47%     | 79.22%     |
| Lung       | 95.36%     | **97.78%** |
| Lymphoma   | 98.81%     | **99.28%** |
| SRBCT      | **99.26%** | 80.54%     |

SVM outperformed k-NN on most datasets. As a standalone method without prior dimensionality reduction, NRO selected relatively large gene subsets, which motivated the hybrid F-NRO and GNR methods that followed.

---

## 📝 Citation

If you use this work, please cite:

```
Alkamli, S.; Alshamlan, H. Evaluating the Nuclear Reaction Optimization (NRO)
Algorithm for Gene Selection in Cancer Classification.
Diagnostics, 2025, 15(7), 927.
```

---

## 📜 License

This repository is provided for **academic and research purposes only**.
