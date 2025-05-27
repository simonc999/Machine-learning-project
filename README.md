├── pima-indians-diabetes.csv  ← data csv
├── pima-indians-diabetes.names    ← original data description
├── pima-indians-diabetes.data    ← original data 
├── relation_EN.ipynb    ← executable notebook (English)
└── relation_IT.ipynb    ← executable notebook (Italian)
``` :contentReference[oaicite:1]{index=1}  

#### 1 · Data handling  
* pandas profiling → custom loaders  
* Null/zero replacement + winsorising  
* Train/validation/test split with *StratifiedShuffleSplit*  

#### 2 · Pre-processing  
* MinMaxScaler 
* **SMOTE** for minority-class up-sampling  
* Feature-selection switch: LR Ranking | Chi2 Ranking | PCA

#### 3 · Model suite  
| Alias | Estimator | Tuned hyper-parameters |
|-------|-----------|------------------------|
| `logreg` | LogisticRegression | penalty, C |
| `knn` | KNeighborsClassifier | k, weights, metric |
| `svm` | SVC | kernel, C, γ |
| `rf` | RandomForestClassifier | n_estimators, max_depth, class_weight |
| `gbc` | GradientBoostingClassifier | n_estimators, learning_rate, subsample |
| `xgb` | XGBClassifier | depth, learning_rate, colsample, eval_metric |

*All models evaluated with **Stratified CV (k=10)**.*

#### 4 · Performances
* IC evaluation at 95%
* Tukey test
* ROC, PR 

