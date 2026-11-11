# Predicting depression from everyday information (NHANES 2017–2018)

NIT6160 Data Warehousing and Mining project: can everyday information (sleep, income, lifestyle) point to who should be screened for depression?

- **Data:** NHANES 2017–2018 (CDC), 5,068 adults, 15 predictors. Target: PHQ-9 score ≥ 10 (9.1% of adults).
- **Models:** logistic regression, random forest and SVM (80/20 stratified split, 5-fold cross-validation, grid search on F1, balanced class weights). Logistic regression is best: recall 0.685, F1 0.324, ROC-AUC 0.759.
- **Clustering:** k-means (k = 3) on five numeric columns finds two lower-income groups (11.9% and 9.6% depressed) and a higher-income group (4.6%).

## Files

| File | What it is |
|---|---|
| `depression_analysis.ipynb` | The full analysis. Run top to bottom; it downloads the 8 NHANES files from the CDC into `data/raw/` and builds `data/nhanes_clean.csv`. |
| `streamlit_app.py` | Presentation app: 18 slides with interactive 3D charts. |
| `notes.json` | Speaker notes shown by the app. |
| `data/nhanes_clean.csv` | The cleaned dataset the app reads. |

## Run the app

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

To deploy on Streamlit Community Cloud, choose this repo, branch `main`, main file `streamlit_app.py`, and Python 3.12.

Author: Kushal Bansal
