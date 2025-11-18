# Crop Recommendation System: Data Analysis & Model Optimization

This repository contains an analysis of crop recommendation datasets, focusing on the comparison between original field data and an enlarged synthetic dataset generated using TVAE (Tabular Variational Autoencoder). The project explores data statistics, trains multiple machine learning classifiers, and performs hyperparameter tuning.

## 📂 Files Overview

### 1. `Enlarged_dataset.ipynb`
**Focus:** Exploratory Data Analysis (EDA) on the Augmented Dataset.

This notebook focuses on analyzing the large-scale synthetic dataset (81,000 samples) to ensure its statistical quality and understand the distribution of agricultural features.

* **Key Features:**
    * **Data Loading:** Imports the `synthetic_crop_data_tvae.csv` containing ~81k records.
    * **Statistical Summary:** Generates global statistics (Mean, Std Dev, Quartiles) for features like Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, pH, and Rainfall.
    * **Crop-wise Analysis:** Uses custom formatting to breakdown feature statistics specifically for each label (e.g., Apple, Banana, Rice), helping identify specific requirements for each crop.
    * **Infrastructure:** Sets up the environment with libraries for data manipulation (`pandas`, `numpy`) and visualization (`seaborn`, `rich`).

### 2. `Original_vs_Synthetic_Updated.ipynb`
**Focus:** Model Benchmarking & Hyperparameter Tuning.

This notebook establishes a robust pipeline to compare machine learning performance between the original dataset (2,200 samples) and the synthetic dataset.

* **Key Features:**
    * **Comparative Training:** Defines a `train_models()` workflow to train 7 classifiers on both datasets simultaneously:
        * Decision Tree, Logistic Regression, Random Forest, LightGBM, SVC, Naive Bayes, and K-Nearest Neighbors.
    * **Evaluation Metrics:** Implements `compare_models()` to visualize side-by-side Confusion Matrices and calculate Accuracy, Precision, Recall, and F1-Scores.
    * **Advanced Tuning (Optuna):**
        * **TPE Sampler:** Used for complex models like Random Forest and LightGBM to efficiently search hyperparameter space.
        * **Bayesian Optimization:** Used for models like Decision Trees and KNN to minimize evaluations while maximizing accuracy.
    * **Results:** Concludes with specific recommendations on which models perform best for this specific tabular data (e.g., finding that Random Forest and LightGBM benefit significantly from Optuna tuning).

---

## 🛠️ Setup & Installation

1.  **Environment:** Ensure you have a Python environment (Python 3.8+) with Jupyter installed.
2.  **Dependencies:** Install the required libraries using pip:

    ```bash
    pip install pandas numpy seaborn matplotlib scikit-learn lightgbm optuna rich
    ```

3.  **Data:**
    * Ensure `Crop_recommendation.csv` (Original) and `synthetic_crop_data_tvae.csv` (Synthetic) are placed in the directory specified in the notebooks (or update the file paths in the first code cell).

## 🚀 Usage

1.  **Step 1: Data validation**
    Run `Enlarged_dataset.ipynb` first to understand the statistical properties of the synthetic data. Review the "Crop-wise Summary Statistics" tables to verify that the synthetic data preserves the biological requirements of the plants (e.g., Rice requiring high rainfall).

2.  **Step 2: Modeling & Comparison**
    Run `Original_vs_Synthetic_Updated.ipynb`.
    * **Training:** The notebook will automatically train all classifiers on both datasets.
    * **Comparison:** Look for the side-by-side plots to see if the synthetic data successfully trains models that generalize well (or if it introduces noise).
    * **Tuning:** The final sections run Optuna trials. This may take time; monitor the "Best Score" logs to see real-time optimization.