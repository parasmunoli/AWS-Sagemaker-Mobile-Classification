# AWS Sagemaker Mobile Classification

This project leverages AWS Sagemaker to build, train, and deploy a machine learning model for classifying mobile devices into various categories based on given features.

---

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [Dataset](#dataset)  
6. [Model Development](#model-development)  
7. [AWS Deployment](#aws-deployment)  
8. [Contributing](#contributing)  
9. [License](#license)

---

## Overview

The **AWS Sagemaker Mobile Classification** project demonstrates end-to-end machine learning workflows, including data preprocessing, model training, and deployment using AWS Sagemaker. The model predicts mobile device categories based on user-provided data.

---

## Features

- **Data Processing:** Handles preprocessing steps for input features.  
- **Model Training:** Includes a Jupyter Notebook for model development.  
- **AWS Deployment:** Integration with AWS Sagemaker for scalable deployment.  
- **Testing:** Validates model performance using provided datasets.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/parasmunoli/AWS-Sagemaker-Mobile-Classification.git
   cd AWS-Sagemaker-Mobile-Classification

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt

## Usage
1. **Model Training:** Use `model.ipynb` to train and evaluate the model.
2. **Run Scripts:** Execute `script.py` to preprocess data or interact with the trained model.
3. **Test Model:** Utilize `test-V-1.csv` to validate predictions.

## Dataset
- Training Data: `train-V-1.csv`
- Test Data: `test-V-1.csv`
The datasets include features relevant to mobile device classification.

## Model Development
- Frameworks Used: Scikit-learn, Pandas, and AWS Sagemaker.
- Steps Covered:
    - Data preprocessing.
    - Model training and evaluation.
    - Exporting the trained model for deployment.