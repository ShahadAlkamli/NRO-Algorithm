# Nuclear Reaction Optimization (NRO) for Gene Selection  
### Official Implementation of the Algorithm Used in the Published Study  
**“Evaluating the Nuclear Reaction Optimization (NRO) Algorithm for Gene Selection in Cancer Classification.”**  
Alkamli & Alshamlan, 2025 (Diagnostics)  

---

## 📌 Overview

This repository contains the implementation of the **Nuclear Reaction Optimization (NRO)** algorithm applied to **gene selection for microarray cancer datasets**.  
NRO is a physics-inspired metaheuristic that simulates **nuclear fission** and **nuclear fusion** processes to explore and refine feature subsets.

This code reproduces the optimization pipeline described in the published work and can be applied to any high-dimensional biological dataset.

---

## 📁 Repository Structure

```
├── Datasets/
│   ├── Colon.arff
│   ├── Leukemia1.arff
│   ├── Leukemia2.arff
│   ├── Lung.arff
│   ├── Lymphoma.arff
│   └── SRBCT.arff
│
├── NRO.py              # Full implementation of NRO algorithm
└── README.md
```

---

## 🔬 Methodology

### **1. Preprocessing**
- Load `.arff` microarray datasets  
- Handle missing values (mean imputation)  
- Normalize features using Z-score  
- Encode class labels numerically  

### **2. Optimization Using NRO**

The algorithm includes:

#### **Nuclear Fission**
- Diversification phase  
- Gaussian perturbation  
- Mutation of solution components  
- Progressive reduction of step size  

#### **Nuclear Fusion**
- Intensification phase  
- Solution ionization  
- Fusion between candidate solutions  
- Lévy flight mechanism to escape local minima  

### **3. Evaluation**
Each candidate feature subset is evaluated with:

- **Support Vector Machine (SVM)**  
- **k-Nearest Neighbors (k-NN)**  
- **Leave-One-Out Cross Validation (LOOCV)**  

The fitness function balances:

- Classification accuracy  
- Number of selected genes  

---

## ▶️ How to Run

### Install dependencies
```bash
pip install numpy pandas scipy scikit-learn
```

### Run the NRO algorithm
```bash
python NRO.py
```

The script will:

- Load datasets  
- Perform optimization  
- Display accuracy and selected gene indices  


---

## 📦 Datasets

Place all `.arff` datasets inside the `Datasets/` directory.  
The following datasets are supported:

- Colon  
- Leukemia 1  
- Leukemia 2  
- Lung  
- Lymphoma  
- SRBCT  


---

## 📝 Citation

If you use this code, please cite:

```
Alkamli, S.; Alshamlan, H. Evaluating the Nuclear Reaction
Optimization (NRO) Algorithm for Gene Selection in Cancer
Classification. Diagnostics, 2025.
```

---

## 📄 License
This code is provided for **research and academic use**.

