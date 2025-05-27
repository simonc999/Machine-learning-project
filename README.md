├── data/
│   └── pima-indians-diabetes.csv
├── relation_EN.ipynb    ← executable notebook / HTML report
└── src/
    ├── pipelines.py     ← preprocessing & model pipelines
    ├── train.py         ← CLI: python train.py --model random_forest
    └── utils.py         ← metrics, plotting, config
``` :contentReference[oaicite:1]{index=1}  

#### 1 · Data handling  
* pandas profiling → custom loaders  
* Null/zero replacement + winsorising  
* Train/validation/test split with *StratifiedShuffleSplit*  

#### 2 · Pre-processing  
* StandardScaler / MinMaxScaler (inside Pipeline)  
* **SMOTE** for minority-class up-sampling  
* Feature-selection switch: VarianceThreshold | RFE | none  

#### 3 · Model suite  
| Alias | Estimator | Tuned hyper-parameters |
|-------|-----------|------------------------|
| `logreg` | LogisticRegression | penalty, C |
| `knn` | KNeighborsClassifier | k, weights, metric |
| `svm` | SVC | kernel, C, γ |
| `rf` | RandomForestClassifier | n_estimators, max_depth, class_weight |
| `gbc` | GradientBoostingClassifier | n_estimators, learning_rate, subsample |
| `xgb` | XGBClassifier | depth, learning_rate, colsample, eval_metric |

*All models evaluated with **Stratified CV (k=10)**. Training script auto-logs metrics to `results/` and serialises the best artefact (`.joblib`).*

#### 4 · Post-hoc explainability  
* SHAP summary & dependence plots  
* Permutation importance  
* Confusion matrix + ROC, PR curves  

#### 5 · Quick-start

```bash
# 1.  Clone & install
pip install -r requirements.txt

# 2.  Train and evaluate Random Forest
python src/train.py --model rf --optimize grid

# 3.  View interactive report
jupyter nbconvert --to html relation_EN.ipynb && open relation_EN.html
