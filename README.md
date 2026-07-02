# 🍷 Wine Quality Prediction

A comprehensive machine learning project implementing both **classification** and **regression models** to predict wine quality. This project showcases advanced machine learning techniques with comparative analysis across 8 different algorithms.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Key Features](#key-features)
- [Model Performance](#model-performance)
- [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [Quick Start Guide](#quick-start-guide)
- [Key Achievements](#key-achievements)
- [Technologies Used](#technologies-used)
- [Results & Visualizations](#results--visualizations)
- [Future Enhancements](#future-enhancements)
- [Author](#author)

---

## 🎯 Project Overview

This project demonstrates the practical application of machine learning in wine quality assessment through two complementary approaches:

### Classification Approach
Categorizing wines into discrete quality classes (Class 0, Class 1, Class 2) using supervised learning algorithms.

### Regression Approach  
Predicting continuous wine quality scores with high precision using advanced regression techniques.

The project adheres to machine learning best practices including proper data preprocessing, model evaluation, cross-algorithm comparison, and detailed performance analysis.

---

## 📊 Dataset Information

### Dataset Overview
| Aspect | Details |
|--------|---------|
| **Source** | scikit-learn Wine Dataset |
| **Total Samples** | 178 wine samples |
| **Training Set** | 124 samples (70%) |
| **Test Set** | 54 samples (30%) |
| **Feature Count** | 13 physicochemical properties |
| **Target Classes** | 3 wine quality classes |

### Features (Physicochemical Properties)
```
1. Alcohol content           8. Proanthocyanins
2. Malic acid                9. Color intensity
3. Ash                       10. Hue
4. Alkalinity of ash         11. OD280/OD315 of diluted wines
5. Magnesium                 12. Proline
6. Total phenols             13. Flavanoids
7. Nonflavanoid phenols
```

---

## ✨ Key Features

### Algorithm Implementation
- ✅ **5 Classification Models** with comprehensive metrics
- ✅ **3+ Regression Models** for continuous predictions
- ✅ **Feature Importance Analysis** from tree-based models
- ✅ **Model Comparison Framework** for objective evaluation

### Data Processing
- ✅ **StandardScaler Normalization** for optimal convergence
- ✅ **Proper Train-Test Splitting** (70-30 ratio)
- ✅ **Data Validation** ensuring quality assurance

### Analysis & Visualization
- ✅ **Confusion Matrices** for classification evaluation
- ✅ **ROC Curves** for multi-class performance comparison
- ✅ **Residual Analysis** for regression diagnostics
- ✅ **Feature Importance Charts** for model interpretability
- ✅ **Actual vs. Predicted Plots** for visual performance assessment

---

## 🏆 Model Performance

### Classification Models

#### Performance Summary Table

| Rank | Model | Accuracy | Weighted F1 | Macro F1 | Status |
|------|-------|----------|-------------|----------|--------|
| 🥇 | Random Forest | **100.00%** | **1.00** | **1.00** | Perfect |
| 🥈 | Logistic Regression | 98.15% | 0.98 | 0.98 | Excellent |
| 🥉 | Support Vector Classifier | 98.15% | 0.98 | 0.98 | Excellent |
| 4️⃣ | K-Nearest Neighbors | 96.30% | 0.96 | 0.96 | Very Good |
| 5️⃣ | Decision Tree | 96.30% | 0.96 | 0.96 | Very Good |

#### Classification Model Details

**🥇 Random Forest Classifier - BEST PERFORMER**
```
Accuracy: 100.00%
Classification Report:
  - Class 0 (Precision: 1.00, Recall: 1.00, F1: 1.00)
  - Class 1 (Precision: 1.00, Recall: 1.00, F1: 1.00)
  - Class 2 (Precision: 1.00, Recall: 1.00, F1: 1.00)
Weighted Average F1-Score: 1.00
Macro Average F1-Score: 1.00
```

**Key Achievement**: Perfect classification on all test samples with zero misclassifications.

---

**🥈 Logistic Regression**
```
Accuracy: 98.15%
Weighted Average F1-Score: 0.98
Macro Average F1-Score: 0.98
Performance: Highly reliable linear classifier
```

---

**🥉 Support Vector Classifier (SVC)**
```
Accuracy: 98.15%
Weighted Average F1-Score: 0.98
Macro Average F1-Score: 0.98
Performance: Excellent non-linear decision boundaries
```

---

**K-Nearest Neighbors (KNN)**
```
Accuracy: 96.30%
Weighted Average F1-Score: 0.96
Macro Average F1-Score: 0.96
Performance: Strong instance-based learning
```

---

**Decision Tree**
```
Accuracy: 96.30%
Weighted Average F1-Score: 0.96
Macro Average F1-Score: 0.96
Performance: Interpretable tree-based approach
```

### Regression Models

#### Performance Summary Table

| Rank | Model | R² Score | MSE | MAE | Status |
|------|-------|----------|-----|-----|--------|
| 🥇 | Support Vector Regressor | **0.9274** | **0.0437** | **0.1459** | Excellent |
| 🥈 | Linear Regression | 0.9008 | 0.0598 | 0.1911 | Very Good |
| 🥉 | Decision Tree Regressor | 0.7541 | 0.1481 | 0.1111 | Good |

#### Regression Model Details

**🥇 Support Vector Regressor - BEST PERFORMER**
```
R² Score: 0.9274 (Explains 92.74% of variance)
Mean Squared Error: 0.0437
Mean Absolute Error: 0.1459
Prediction Range: High accuracy with minimal deviation
```

**Key Achievement**: Highest accuracy regression model with superior generalization capability.

---

**🥈 Linear Regression**
```
R² Score: 0.9008 (Explains 90.08% of variance)
Mean Squared Error: 0.0598
Mean Absolute Error: 0.1911
Performance: Excellent baseline with strong linear relationships
```

---

**🥉 Decision Tree Regressor**
```
R² Score: 0.7541 (Explains 75.41% of variance)
Mean Squared Error: 0.1481
Mean Absolute Error: 0.1111
Performance: Acceptable with interpretable decision rules
```

---

## 🚀 Installation & Setup

### Prerequisites
```
Python 3.7 or higher
pip package manager
Jupyter Notebook
```

### Step 1: Clone the Repository
```bash
git clone https://github.com/rki123/Wine-quality.git
cd Wine-quality
```

### Step 2: Install Required Libraries
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

**Alternative (using requirements.txt):**
```bash
pip install -r requirements.txt
```

### Step 3: Launch Jupyter Notebook
```bash
jupyter notebook
```

### Step 4: Open and Run the Notebook
```
Open: Ritik_project_Wine_Quality.ipynb
Run: All cells sequentially (Kernel → Restart & Run All)
```

---

## 📁 Project Structure

```
Wine-quality/
│
├── README.md                           # Project documentation (this file)
│
└── Ritik_project_Wine_Quality.ipynb    # Main Jupyter Notebook
    │
    ├── 📦 SECTION 1: Data Loading & Preprocessing
    │   ├── Import libraries
    │   ├── Load wine dataset
    │   ├── Train-test split (70-30)
    │   └── Feature scaling with StandardScaler
    │
    ├── 🔍 SECTION 2: Classification Models
    │   ├── Model 1: Logistic Regression
    │   ├── Model 2: Support Vector Classifier
    │   ├── Model 3: Decision Tree Classifier
    │   ├── Model 4: Random Forest Classifier ⭐
    │   └── Model 5: K-Nearest Neighbors
    │       └── Confusion Matrices
    │       └── Classification Reports
    │       └── Feature Importance Analysis
    │
    ├── 📈 SECTION 3: Regression Models
    │   ├── Model 1: Linear Regression
    │   ├── Model 2: Support Vector Regressor ⭐
    │   └── Model 3: Decision Tree Regressor
    │       └── Actual vs. Predicted Plots
    │       └── Residual Analysis
    │       └── Error Metrics
    │
    └── 📊 SECTION 4: Comparative Analysis & Visualizations
        ├── Accuracy comparison charts
        ├── F1-Score comparison charts
        ├── ROC curves for all classifiers
        └── Performance metrics dashboard
```

---

## 💻 Quick Start Guide

### Example 1: Train Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize and train model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")  # Output: 1.0000
print(classification_report(y_test, y_pred, target_names=class_names))
```

---

### Example 2: Train Support Vector Regressor

```python
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

# Initialize and train model
svr_model = SVR()
svr_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svr_model.predict(X_test_scaled)

# Evaluate
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R² Score: {r2:.4f}")    # Output: 0.9274
print(f"MSE: {mse:.4f}")        # Output: 0.0437
```

---

### Example 3: Feature Scaling (Essential!)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 🎖️ Key Achievements

### 🏅 Classification Excellence
- ✨ **Perfect Classification**: Random Forest achieved **100% accuracy** with zero errors
- 📈 **High Consistency**: 3 out of 5 models exceeded **98% accuracy**
- 🎯 **Balanced Performance**: Maintained excellent precision-recall across all classes
- 🔬 **Robust Validation**: All models showed strong generalization capability

### 📊 Regression Excellence
- 🥇 **Superior SVR Performance**: Achieved **0.9274 R² score** (92.74% variance explained)
- 📉 **Minimal Error**: Average prediction error of only **0.1459** units
- 📋 **Strong Baseline**: Linear Regression also achieved excellent **0.9008 R²** score
- 🎓 **Multiple Model Validation**: Regression tested across 3 different approaches

### 🔬 Project Excellence
- 📚 **Comprehensive Evaluation**: 8 different machine learning algorithms analyzed
- 📊 **Advanced Visualizations**: 
  - Confusion matrices for multi-class analysis
  - ROC curves for classification comparison
  - Feature importance charts for interpretability
  - Residual plots for regression diagnostics
- 🛠️ **Professional Implementation**: 
  - Proper data preprocessing pipeline
  - Train-test validation strategy
  - Reproducible results with fixed random seeds
  - Clean, documented code structure
- 🎯 **Actionable Insights**: Clear recommendations for model selection based on use case

---

## 🛠️ Technologies Used

### Core Libraries
| Technology | Purpose | Version |
|-----------|---------|---------|
| **Python** | Programming Language | 3.7+ |
| **NumPy** | Numerical Computing | Latest |
| **Pandas** | Data Manipulation | Latest |
| **scikit-learn** | Machine Learning | Latest |

### Visualization & Analysis
| Technology | Purpose |
|-----------|---------|
| **Matplotlib** | Plot generation and visualization |
| **Seaborn** | Statistical data visualization |
| **Jupyter Notebook** | Interactive development environment |

### Computing Environment
| Component | Details |
|-----------|---------|
| **Platform** | Google Colab |
| **GPU** | NVIDIA T4 (for acceleration) |
| **RAM** | Sufficient for dataset operations |

---

## 📊 Results & Visualizations

### Classification Visualizations Generated
```
✓ Confusion Matrix (5 models)
  - Visual representation of True Positives, False Positives, etc.
  - Color-coded heatmaps for easy interpretation
  - Per-class accuracy metrics

✓ ROC Curves
  - Multi-class comparison across all classifiers
  - AUC (Area Under Curve) scoring
  - Diagonal baseline reference

✓ Performance Comparison Charts
  - Accuracy bar plots
  - Weighted F1-Score comparison
  - Macro F1-Score comparison
```

### Regression Visualizations Generated
```
✓ Actual vs. Predicted Plots
  - Scatter plots for each regression model
  - Perfect prediction line reference
  - Deviation visualization

✓ Residual Plots
  - Residual distribution analysis
  - Error pattern detection
  - Heteroscedasticity assessment

✓ Feature Importance Charts
  - Bar plots showing feature contribution
  - Ranked by importance for Decision Tree models
  - Top contributing features highlighted
```

### Decision Tree Visualization
```
✓ Tree Structure Diagrams
  - Visual representation of decision rules
  - Node colors based on class prediction
  - Feature and threshold information at each split
```

---

## 🔮 Future Enhancements

### Model Improvement
- [ ] **Hyperparameter Tuning** using GridSearchCV and RandomizedSearchCV
- [ ] **Cross-Validation** for more robust performance estimation
- [ ] **Ensemble Methods** combining multiple models for better accuracy
- [ ] **Feature Engineering** to create new predictive features
- [ ] **Feature Selection** techniques to reduce dimensionality

### Data & Model Expansion
- [ ] Integration with **real-world wine quality datasets**
- [ ] **Class Imbalance Handling** using SMOTE or class weighting
- [ ] **Outlier Detection** and treatment techniques
- [ ] **Normalization Alternatives** (MinMaxScaler, RobustScaler)

### Explainability & Deployment
- [ ] **SHAP Values** for model interpretation
- [ ] **LIME (Local Interpretable Model-agnostic Explanations)**
- [ ] **Flask/Django Web Application** for model deployment
- [ ] **REST API** for predictions
- [ ] **Model Persistence** using joblib or pickle

### Advanced Techniques
- [ ] **Neural Networks** (Deep Learning) comparison
- [ ] **Gradient Boosting** (XGBoost, LightGBM) models
- [ ] **AutoML** frameworks for automatic model selection
- [ ] **Time Series Analysis** if temporal data available

---

## 📝 Author

**Ritik Kumar**  
GitHub: [@rki123](https://github.com/rki123)  
Project Repository: [Wine-quality](https://github.com/rki123/Wine-quality)

---


## 🙏 Acknowledgments

- **scikit-learn** for the Wine dataset and machine learning algorithms
- **Matplotlib & Seaborn** for visualization capabilities
- **Google Colab** for computational resources
- Open-source machine learning community for best practices and guidance

## 📈 Performance Summary at a Glance

```
CLASSIFICATION:  ✅ 100% accuracy achieved (Random Forest)
REGRESSION:      ✅ 0.9274 R² score achieved (Support Vector Regressor)
ALGORITHMS:      ✅ 8 models evaluated and compared
DOCUMENTATION:   ✅ Comprehensive with visualizations
```

---




---
