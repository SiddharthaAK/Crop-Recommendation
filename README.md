# Crop Recommendation and Synthetic Data Analysis

This repository contains the code and resources for a machine learning project focused on crop recommendation. The primary objective is to evaluate and compare the performance of various classification algorithms on an original agricultural dataset versus an enlarged dataset augmented with synthetic data.

## Project Overview

Precision agriculture relies on accurate data to recommend the most suitable crops for specific soil and climatic conditions. This project investigates the utility of synthetic data generation (specifically using TVAE) in machine learning pipelines. It includes:

* Exploratory Data Analysis (EDA) of agricultural features.
* Training and tuning multiple machine learning classifiers.
* Comparative analysis of model performance (accuracy, precision, recall, F1-score) on original versus synthetic datasets.
* Resource usage monitoring (CPU and Memory) during training.

## Datasets

The project utilizes two primary data sources:

1.  **Original Dataset** (`Crop_recommendation.csv`): Contains real-world agricultural data.
2.  **Synthetic Dataset** (`synthetic_crop_data_tvae.csv`): An augmented dataset generated to increase sample size and diversity while maintaining statistical properties of the original data.

### Features
* **N**: Ratio of Nitrogen content in soil
* **P**: Ratio of Phosphorous content in soil
* **K**: Ratio of Potassium content in soil
* **temperature**: Temperature in degree Celsius
* **humidity**: Relative humidity in %
* **ph**: pH value of the soil
* **rainfall**: Rainfall in mm
* **label**: The target crop class

## Requirements

To run the code in this repository, you need Python 3.x and the following libraries installed:

* pandas
* numpy
* scikit-learn
* lightgbm
* matplotlib
* seaborn
* rich

## Installation

1.  Clone this repository to your local machine.
2.  Ensure you have Python installed.
3.  Install the required dependencies using pip:

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn rich
```
## Usage

### Getting Started

1. Navigate to the repository folder in your terminal or command prompt.

2. Start the Jupyter Notebook server:
```bash
   jupyter notebook
```

3. Open and run the following notebooks:

   - **Original_vs_Synthetic_Updated.ipynb**: This is the main analysis notebook. It loads both datasets, performs comparative training of models, visualizes results (Confusion Matrices, Performance Plots), and conducts hyperparameter tuning.
   
   - **Enlarged_dataset.ipynb**: This notebook focuses on the larger synthetic dataset, detailing data loading, feature analysis, and initial model testing.

4. Execute the cells sequentially to reproduce the analysis, training, and evaluation steps.

## Models Implemented

The project evaluates the following classification algorithms:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- LightGBM Classifier
- Support Vector Classifier (SVC)
- Gaussian Naive Bayes
- K-Nearest Neighbors (KNN)

## Methodology

### 1. Data Loading & Preprocessing
The data is loaded using Pandas. The datasets are split into training and testing sets.

### 2. Model Training
Each of the seven classifiers is initialized and fitted to the training data.

### 3. Evaluation
Models are evaluated on the test set using standard metrics:
- Accuracy
- Precision, Recall, and F1-Score (weighted averages)
- Confusion Matrices

### 4. Hyperparameter Tuning
`RandomizedSearchCV` is used to find optimal parameters for models like LightGBM, Decision Tree, Random Forest, and SVC.

### 5. Comparison
The results from the original dataset are compared against the synthetic dataset to assess the quality and utility of the synthetic data generation.

## Results

The analysis generates detailed classification reports and visual comparisons. Refer to the output cells in the notebooks for specific metric scores, execution times, and resource utilization graphs.
