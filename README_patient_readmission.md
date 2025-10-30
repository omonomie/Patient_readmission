# Hospital Patient Readmission Prediction Project

## 📋 Project Overview

This project implements a complete machine learning pipeline for predicting 30-day hospital readmissions using R and the tidymodels framework. It serves as a comprehensive case study and minor project for the ICSA9PR4 course, demonstrating end-to-end data science workflow from data collection to model deployment.

### 🎯 Objectives

- Predict 30-day hospital readmission risk for diabetic patients
- Identify critical factors contributing to readmission
- Compare multiple machine learning algorithms (Logistic Regression, Random Forest, XGBoost)
- Develop an interpretable and clinically actionable predictive model
- Provide a reproducible workflow using modern R packages

## 📊 Dataset Information

**Dataset:** Diabetes 130-US Hospitals (1999-2008)  
**Source:** UCI Machine Learning Repository  
**URL:** https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

### Dataset Characteristics

- **Records:** 101,766 patient encounters
- **Features:** 50+ variables including:
  - Demographics (age, gender, race)
  - Admission details (admission type, discharge disposition)
  - Clinical measurements (lab procedures, medications, diagnoses)
  - Diabetes-specific information (HbA1c, glucose levels)
- **Target Variable:** 30-day readmission status
- **Time Period:** 10 years (1999-2008) across 130 US hospitals

## 🛠️ Technical Stack

### Core Technologies

- **R Version:** 4.5.1 or higher
- **Primary Framework:** tidymodels (version 1.2.0+)

### Required R Packages

```r
# Core tidymodels ecosystem
library(tidymodels)    # Meta-package (includes recipes, parsnip, workflows, tune, yardstick)
library(tidyverse)     # Data manipulation and visualization

# Machine learning engines
library(ranger)        # Random Forest implementation
library(xgboost)       # Gradient Boosting implementation
library(glmnet)        # Regularized regression

# Additional utilities
library(themis)        # Class imbalance handling (SMOTE)
library(vip)           # Variable importance plots
library(skimr)         # Data exploration
library(corrplot)      # Correlation visualization
library(patchwork)     # Plot composition
library(doParallel)    # Parallel processing
```

### Installation

```r
# Install tidymodels and dependencies
install.packages("tidymodels")

# Install machine learning engines
install.packages(c("ranger", "xgboost", "glmnet"))

# Install utility packages
install.packages(c("themis", "vip", "skimr", "corrplot", "patchwork", "doParallel"))
```

## 📁 Project Structure

```
patient-readmission-prediction/
│
├── patient_readmission_prediction.qmd    # Main analysis file (Quarto document)
├── README.md                             # This file
│
├── data/                                 # Data directory (created automatically)
│   ├── diabetic_data.csv                # Raw dataset (downloaded from UCI)
│   └── IDs_mapping.csv                  # Variable mappings
│
├── models/                               # Saved models (created during execution)
│   ├── readmission_model.rds            # Final trained model
│   └── model_metadata.rds               # Model performance metrics
│
└── figures/                              # Generated visualizations
    ├── eda_plots/                       # Exploratory data analysis
    ├── model_performance/               # Model evaluation charts
    └── feature_importance/              # Variable importance plots
```

## 🚀 Getting Started

### Step 1: Clone or Download Project

Download the `patient_readmission_prediction.qmd` file to your working directory.

### Step 2: Install Dependencies

Open R or RStudio and run:

```r
# Install required packages
install.packages(c("tidymodels", "ranger", "xgboost", "glmnet", 
                   "themis", "vip", "skimr", "corrplot", "patchwork"))
```

### Step 3: Run the Analysis

**Option A: Using RStudio (Recommended)**

1. Open `patient_readmission_prediction.qmd` in RStudio
2. Click "Render" button (or press Ctrl/Cmd + Shift + K)
3. The analysis will execute and generate an HTML report

**Option B: Using Command Line**

```r
# In R console
quarto::quarto_render("patient_readmission_prediction.qmd")
```

**Option C: Execute Code Chunks Interactively**

1. Open the QMD file in RStudio
2. Run each code chunk individually using Ctrl/Cmd + Shift + Enter
3. This allows step-by-step exploration

### Step 4: Review Output

The rendered HTML report will be created in the same directory:
- `patient_readmission_prediction.html` - Complete analysis report

## 📖 Project Workflow

### 1. Data Collection and Loading
- Automatic download from UCI repository
- Data structure inspection
- Initial quality checks

### 2. Exploratory Data Analysis (EDA)
- Target variable distribution analysis
- Missing data assessment
- Feature distributions and correlations
- Bivariate analysis of readmission patterns

### 3. Data Preprocessing
- Binary target variable creation (30-day readmission: Yes/No)
- Feature selection based on domain knowledge
- ICD-9 diagnosis code grouping into clinical categories
- Handling of categorical and numerical variables

### 4. Data Splitting
- 75/25 train-test stratified split
- 5-fold cross-validation with stratification
- Maintains class distribution across splits

### 5. Feature Engineering (Recipes)
- Zero-variance and near-zero variance filtering
- Dummy variable encoding for categorical features
- Normalization of numerical predictors
- SMOTE for class imbalance handling

### 6. Model Specification
Three algorithms evaluated:
- **Logistic Regression** (with elastic net regularization)
- **Random Forest** (ranger engine)
- **XGBoost** (gradient boosting)

### 7. Hyperparameter Tuning
- Grid search for logistic regression
- Latin hypercube sampling for XGBoost efficiency
- 5-fold cross-validation for all models
- Multiple metrics: ROC AUC, accuracy, sensitivity, specificity

### 8. Model Selection
- Performance comparison across algorithms
- Selection based on cross-validated ROC AUC
- Consideration of interpretability and computational efficiency

### 9. Final Model Evaluation
- Training on full training set
- Comprehensive testing on held-out test set
- Confusion matrix analysis
- ROC and Precision-Recall curves
- Clinical interpretation of results

### 10. Feature Importance Analysis
- Variable importance extraction
- Top predictors identification
- Clinical relevance assessment

### 11. Model Deployment
- Model serialization (RDS format)
- Prediction function for new patients
- Metadata storage for model tracking

## 📊 Key Features

### 1. Comprehensive Data Pipeline
✅ Automatic data download and preprocessing  
✅ Missing data handling strategies  
✅ Clinical domain-specific feature engineering  
✅ Reproducible workflow with seed control

### 2. Advanced Machine Learning
✅ Multiple algorithm comparison  
✅ Systematic hyperparameter tuning  
✅ Cross-validation for robust estimates  
✅ Class imbalance handling (SMOTE)

### 3. Rigorous Evaluation
✅ Multiple performance metrics  
✅ ROC and PR curves  
✅ Confusion matrix analysis  
✅ Feature importance interpretation

### 4. Clinical Relevance
✅ Interpretable results  
✅ Actionable insights for healthcare providers  
✅ Discussion of clinical implications  
✅ Recommendations for intervention strategies

### 5. Production-Ready
✅ Model saving and versioning  
✅ Prediction function for deployment  
✅ Metadata tracking  
✅ Documentation and reproducibility

## 📈 Expected Results

Based on literature and dataset characteristics:

- **ROC AUC:** 0.62 - 0.68 (Good discrimination)
- **Accuracy:** 85% - 92% (High overall correct predictions)
- **Sensitivity:** 40% - 60% (Moderate recall of readmissions)
- **Specificity:** 85% - 95% (High identification of non-readmissions)

**Note:** Performance may vary based on:
- Hyperparameter tuning results
- Cross-validation fold variability
- Class imbalance handling approach
- Feature engineering choices

## 🎓 Learning Outcomes

By completing this project, students will demonstrate:

1. **Data Science Workflow Mastery**
   - End-to-end pipeline development
   - Reproducible research practices
   - Version control and documentation

2. **R Programming Proficiency**
   - tidymodels framework expertise
   - Modern R package ecosystem
   - Functional programming patterns

3. **Machine Learning Skills**
   - Algorithm selection and comparison
   - Hyperparameter optimization
   - Model evaluation and interpretation

4. **Domain Knowledge Application**
   - Healthcare analytics context
   - Clinical feature engineering
   - Actionable insight generation

5. **Communication Abilities**
   - Technical report writing
   - Data visualization
   - Stakeholder-oriented presentation

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue 1: Dataset Download Fails

**Solution:**
```r
# Manual download option
# 1. Visit: https://archive.ics.uci.edu/dataset/296
# 2. Download the dataset manually
# 3. Extract to data/ folder
# 4. Ensure file is named: diabetic_data.csv
```

#### Issue 2: Package Installation Errors

**Solution:**
```r
# Update R to latest version
# Update RStudio to latest version
# Install packages from CRAN with dependencies
install.packages("tidymodels", dependencies = TRUE)
```

#### Issue 3: Memory Issues During Model Training

**Solution:**
```r
# Reduce grid search size
# Decrease cross-validation folds from 5 to 3
# Use smaller training sample for initial testing

# Example: Sample data for testing
train_sample <- train_data %>% slice_sample(n = 10000)
```

#### Issue 4: Long Computation Time

**Solution:**
```r
# Enable parallel processing
library(doParallel)
cl <- makePSOCKcluster(parallel::detectCores() - 1)
registerDoParallel(cl)

# Reduce hyperparameter grid size
# Use fewer trees for Random Forest (e.g., 100 instead of 500)
# Reduce XGBoost tuning grid (e.g., 10 combinations instead of 30)
```

## 📚 Additional Resources

### Tidymodels Documentation
- Official Website: https://www.tidymodels.org/
- Book: "Tidy Modeling with R" - https://www.tmwr.org/
- Tutorials: https://www.tidymodels.org/start/

### Healthcare Analytics
- CMS Hospital Readmissions Program: https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/hospital-readmissions-reduction-program-hrrp
- Original Research: Strack et al. (2014) "Impact of HbA1c Measurement on Hospital Readmission Rates"

### Machine Learning Best Practices
- scikit-learn documentation (conceptual parallels): https://scikit-learn.org/
- Feature Engineering Book: "Feature Engineering and Selection" by Kuhn & Johnson
- Model Evaluation: "Applied Predictive Modeling" by Kuhn & Johnson

## 👥 Contributors and Credits

**Project Type:** Academic Case Study - Course ICSA9PR4  
**Framework:** tidymodels ecosystem  
**Dataset Source:** UCI Machine Learning Repository  
**Original Research:** Strack et al. (2014)

## 📝 Citation

If you use this project or adapt this code, please cite:

```
Hospital Patient Readmission Prediction using Tidymodels
Case Study and Minor Project - ICSA9PR4
Dataset: Diabetes 130-US Hospitals (UCI ML Repository)
Year: 2025
```

## ⚖️ License and Usage

This project is created for educational purposes under the ICSA9PR4 course requirements.

- **Dataset:** UCI ML Repository (publicly available for research and education)
- **Code:** Available for academic and educational use
- **Modifications:** Encouraged for learning purposes

## 🔄 Version History

- **v1.0 (2025-01-XX):** Initial comprehensive pipeline
  - Complete data preprocessing
  - Three ML algorithms implemented
  - Cross-validation and hyperparameter tuning
  - Model evaluation and interpretation
  - Deployment-ready prediction function

## 🤝 Support and Feedback

For questions, issues, or improvements:

1. Review the troubleshooting section above
2. Check tidymodels documentation: https://www.tidymodels.org/
3. Consult course materials and instructors
4. Review R and RStudio community forums

## 🎯 Next Steps for Extension

Consider these enhancements for advanced projects:

1. **Feature Engineering**
   - Add interaction terms between key variables
   - Create polynomial features
   - Incorporate temporal patterns

2. **Advanced Models**
   - Deep learning with keras/tensorflow
   - Ensemble stacking methods
   - Bayesian approaches

3. **Fairness Analysis**
   - Evaluate model equity across demographic groups
   - Implement fairness constraints
   - Analyze disparate impact

4. **Temporal Validation**
   - Time-based train-test splits
   - Prospective validation study
   - Concept drift analysis

5. **Deployment**
   - Shiny web application interface
   - REST API with plumber
   - Integration with hospital EHR systems

---

**Last Updated:** October 2025  
**Status:** Complete and ready for execution  
**Estimated Runtime:** 15-30 minutes (depending on hardware)

**🚀 Ready to get started? Open the QMD file and begin your analysis!**
