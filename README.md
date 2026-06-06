# 💳 Credit Card Fraud Detection
A machine learning project that trains a **Random Forest classifier** on a real-world, anonymised credit card transaction dataset to detect fraudulent transactions, with a focus on handling severe class imbalance and optimising for fraud-relevant evaluation metrics.

<img width="625" height="482" alt="image" src="https://github.com/user-attachments/assets/9e1c4cc2-6917-4338-bbbb-7d93bbd5b448" />

---

## 📂 Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **File:** `creditcard.csv` (~150 MB, not included in this repo due to size)
- **Shape:** 284,807 transactions × 31 columns
- **Class distribution:** Highly imbalanced — fraudulent transactions represent ~0.17% of all records

| Column | Description |
|---|---|
| `Time` | Seconds elapsed since the first transaction in the dataset |
| `V1–V28` | PCA-transformed anonymised features |
| `Amount` | Transaction amount (original scale) |
| `Class` | Target label — `0` = Legitimate, `1` = Fraud |

> **Note:** The V1–V28 features are the result of a PCA transformation applied by the dataset authors to protect cardholder confidentiality. Original feature names are not available.

---

## 🔬 Project Pipeline

### 1. Data Ingestion & Transformation

- Loaded the CSV from Google Drive into a pandas DataFrame
- Audited 'null' counts and data types
- **Time → Hour:** Converted raw seconds into a 0–23 hour-of-day feature (`(Time // 3600) % 24`) to capture temporal transaction patterns
- **Amount → Log_Amount:** Applied `log1p` to compress the heavily right-skewed amount distribution
- Dropped the original `Time` and `Amount` columns after transformation
- Applied `StandardScaler` to `Hour` and `Log_Amount` so they align with the zero-mean, unit-variance characteristics of the PCA-derived V columns

### 2. Exploratory Data Analysis

- Computed per-feature mean values for fraud vs. legitimate classes across all V1–V28 columns
- Ranked features by **absolute mean difference** to identify the strongest fraud signals
- Visualised the top 5 discriminating features using side-by-side boxplots to confirm distribution shifts between classes

### 3. Modelling

- Defined features (`X`) and target (`y = Class`)
- Performed a **stratified 70/30 train-test split** (`random_state=25`) to preserve class proportions in both sets
- Trained a **Random Forest Classifier** with the following configuration:

```python
RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',   # compensates for class imbalance
    max_depth=20,
    random_state=25,
    n_jobs=-1
)
```

### 4. Evaluation

| Metric | Detail |
|---|---|
| Classification Report | Precision, Recall, F1 per class |
| Confusion Matrix | Visualised with `ConfusionMatrixDisplay` |
| Feature Importance | Top 10 features ranked by Gini importance |
| **PR-AUC** | Primary evaluation metric — scored **> 0.85** |

> **Why PR-AUC and not ROC-AUC?**  
> With a class imbalance of ~0.17%, ROC-AUC is an optimistic metric — a model that catches most fraud but floods operations with false positives can still score well. Precision-Recall AUC is far more informative because it focuses on how well the model ranks true fraud above noise, without rewarding easy negatives.

---

## 📊 Results

- **PR-AUC > 0.85** — the model ranks fraudulent transactions significantly higher than legitimate ones while keeping false positives manageable
- Feature importance analysis surfaced the V-columns with the largest mean-difference from EDA as the most predictive, validating the exploratory step
- `class_weight='balanced'` proved effective at correcting for the skewed class distribution without requiring resampling

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical transformations |
| `matplotlib` / `seaborn` | Visualisations |
| `scikit-learn` | Modelling, evaluation, and preprocessing |

---

## 🚀 Getting Started

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. **Download the dataset**  
   Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in your working directory (or update the file path in the notebook).

4. **Run the notebook**  
   Open `credit_card_fraud_detection.ipynb` in Jupyter or Google Colab and run all cells.

---

## 💡 Key Takeaways

- Raw temporal and monetary features require domain-informed engineering even when most features are already PCA-transformed
- `class_weight='balanced'` is a lightweight but effective first response to class imbalance in tree-based models
- **PR-AUC is the right north-star metric** for fraud detection — optimising for it aligns model performance with real-world operational cost (false positives = analyst workload; false negatives = financial loss)

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).

---

## 🙋 Author

**Charchit**  
Fraud Analytics, Operations & Strategy | 4 years across banking, fintech & aviation  

[LinkedIn](https://linkedin.com/in/charchit46) · [GitHub](https://github.com/0xsquawk)
