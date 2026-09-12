# XGBoost — Complete Revision Notes
> Covers: Concept + Theory + Hyperparameters + Code + Pipeline + Interview Q&A

---

## 1. What is XGBoost?

**Full name:** eXtreme Gradient Boosting

XGBoost is an **ensemble method** — it combines many weak models (decision trees) into one strong model.

```
Weak model  = one small decision tree (not very accurate alone)
Ensemble    = 100-1000 such trees combined = very accurate
```

**Family tree:**
```
Ensemble Methods
├── Bagging          → Random Forest (trees built in PARALLEL, independent)
└── Boosting         → AdaBoost → Gradient Boosting → XGBoost
                       (trees built SEQUENTIALLY, each fixing previous errors)
```

---

## 2. How XGBoost Works (Core Concept)

### Step-by-step intuition:

```
Iteration 1: Tree 1 makes predictions → some right, some wrong
             Wrong predictions = residuals (errors)

Iteration 2: Tree 2 tries to predict the RESIDUALS of Tree 1
             (it focuses on where Tree 1 was wrong)

Iteration 3: Tree 3 predicts residuals of Tree 1 + Tree 2
             ...and so on for n_estimators trees

Final prediction = sum of all trees × learning_rate
```

### Example:
```
Real value:       10
Tree 1 predicts:   7  → error = 3
Tree 2 predicts:   2  → error = 1  (corrects most of Tree 1's error)
Tree 3 predicts:  0.7 → error = 0.3
...

Final = 7 + 2 + 0.7 + ... ≈ 10
```

### Key difference from Random Forest:
```
Random Forest:   100 trees trained INDEPENDENTLY, vote/average
XGBoost:         100 trees trained SEQUENTIALLY, each fixes previous errors
Result:          XGBoost almost always outperforms Random Forest
```

---

## 3. Key Hyperparameters — What Each Does

### Tree Structure
| Parameter | What it controls | Good starting value | Effect of too high |
|---|---|---|---|
| `n_estimators` | Number of trees | 100-300 | Overfitting (use early stopping) |
| `max_depth` | Max depth of each tree | 3-6 | Overfitting |
| `min_child_weight` | Min samples in leaf | 1 | Overfitting if too low |

### Learning
| Parameter | What it controls | Good starting value | Note |
|---|---|---|---|
| `learning_rate` (eta) | How much each tree contributes | 0.01-0.1 | Lower = need more trees but better |
| `n_estimators` | Works with learning_rate | Higher when lr is lower | Use early stopping |

**Rule:** Low learning_rate + high n_estimators + early_stopping = best results

### Randomness (prevents overfitting)
| Parameter | What it controls | Good starting value |
|---|---|---|
| `subsample` | % of rows used per tree | 0.6-1.0 |
| `colsample_bytree` | % of features used per tree | 0.6-1.0 |

### Regularisation
| Parameter | What it controls | Good starting value |
|---|---|---|
| `reg_alpha` | L1 regularisation (makes weights sparse) | 0 |
| `reg_lambda` | L2 regularisation (shrinks weights) | 1 |
| `scale_pos_weight` | For imbalanced data: neg_count/pos_count | neg/pos ratio |

### Objective (task type)
```python
'binary:logistic'     # Binary classification (default for XGBClassifier)
'multi:softmax'       # Multi-class classification
'reg:squarederror'    # Regression (default for XGBRegressor)
```

---

## 4. Early Stopping

**Problem without early stopping:** You guess n_estimators. Too high = overfit. Too low = underfit.

**With early stopping:** Training stops automatically when validation loss stops improving.

```python
model = XGBClassifier(
    n_estimators=1000,           # set high — early stopping will stop it
    learning_rate=0.05,
    early_stopping_rounds=50,    # stop if no improvement for 50 rounds
    eval_metric='logloss'
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)
print(f"Best iteration: {model.best_iteration}")   # actual trees used
```

---

## 5. Feature Importance — 3 Types

```python
from xgboost import plot_importance

# Type 1: weight — how many times feature used in splits
# Most common, but biased toward high-cardinality features

# Type 2: gain — avg improvement in loss from splits on this feature
# BEST for understanding true importance

# Type 3: cover — avg number of samples affected by splits on this feature

plot_importance(model, importance_type='gain')
```

**In interviews:** Always say "I use gain importance, not weight, because weight is biased toward features with many unique values."

---

## 6. SHAP Values (Explain Predictions)

```python
import shap

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot 1: which features matter most OVERALL?
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Plot 2: WHY did THIS specific prediction happen?
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0],
    feature_names=feature_names
))
```

**What SHAP tells you:**
```
Positive SHAP value → feature pushed prediction HIGHER (toward class 1)
Negative SHAP value → feature pushed prediction LOWER (toward class 0)
Magnitude           → how much it pushed
```

---

## 7. Cross Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
print(f"Mean AUC: {scores.mean():.4f} ± {scores.std():.4f}")

# Why: single train/test split can be lucky
# 5-fold CV = train+evaluate on 5 different splits, average the result
# More reliable estimate of real performance
```

---

## 8. XGBoost in a Pipeline

**What is Pipeline?**
Pipeline chains preprocessing steps + model together so nothing is forgotten in production.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Preprocessing for numeric columns
num_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale',  StandardScaler())
])

# Preprocessing for categorical columns
cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessors
preprocessor = ColumnTransformer([
    ('num', num_pipe, ['age', 'fare', 'pclass']),
    ('cat', cat_pipe, ['sex', 'embarked'])
])

# Full pipeline: preprocessing + XGBoost
full_pipeline = Pipeline([
    ('prep',  preprocessor),
    ('model', XGBClassifier(n_estimators=200, learning_rate=0.05,
                             max_depth=4, random_state=42,
                             eval_metric='logloss'))
])

# ONE call trains everything
full_pipeline.fit(X_train, y_train)

# ONE call predicts (preprocessing happens automatically)
y_pred = full_pipeline.predict(X_test)

# Tune hyperparameters with double underscore syntax
from sklearn.model_selection import GridSearchCV
param_grid = {
    'model__n_estimators':  [100, 200],
    'model__max_depth':     [3, 4, 6],
    'model__learning_rate': [0.05, 0.1],
}
grid = GridSearchCV(full_pipeline, param_grid, cv=5, scoring='roc_auc')
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)

# Save entire pipeline (preprocessing + model together)
import joblib
joblib.dump(full_pipeline, 'xgboost_pipeline.pkl')

# Load and predict on raw new data
loaded = joblib.load('xgboost_pipeline.pkl')
import pandas as pd
new_data = pd.DataFrame([{'age':25, 'fare':50, 'pclass':1,
                           'sex':'female', 'embarked':'C'}])
print(loaded.predict(new_data))   # preprocessing happens automatically
```

**Why Pipeline matters:**
```
Without Pipeline:  Must manually scale/encode in both training AND prediction
                   Easy to forget → bug in production

With Pipeline:     One .predict() call handles everything
                   Impossible to forget a preprocessing step
```

---

## 9. Handle Imbalanced Data

```python
# Check balance
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
print(f"Negative: {neg}, Positive: {pos}, Ratio: {neg/pos:.2f}")

# XGBoost built-in solution
model = XGBClassifier(
    scale_pos_weight = neg/pos,    # tells XGBoost to weight minority class higher
    ...
)

# Also: adjust threshold (default 0.5)
y_prob = model.predict_proba(X_test)[:,1]
y_pred_adjusted = (y_prob >= 0.3).astype(int)  # lower threshold = catch more positives
```

---

## 10. Save and Load Model

```python
import joblib

# Save
joblib.dump(model, 'xgboost_model.pkl')

# Load
model = joblib.load('xgboost_model.pkl')
prediction = model.predict(new_data)
```

---

## 11. All 3 Modes — Quick Reference

```python
# Binary Classification
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=200, eval_metric='logloss')
model.fit(X_train, y_train)
model.predict(X_test)          # → 0 or 1
model.predict_proba(X_test)    # → [[0.3, 0.7], ...]

# Regression
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=200, eval_metric='rmse')
model.fit(X_train, y_train)
model.predict(X_test)          # → [12.5, 8.3, 22.1, ...]

# Multi-class
from xgboost import XGBClassifier
model = XGBClassifier(objective='multi:softmax', num_class=3,
                       eval_metric='mlogloss')
model.fit(X_train, y_train)
model.predict(X_test)          # → 0, 1, 2 ...
model.predict_proba(X_test)    # → [[0.1, 0.7, 0.2], ...]
```

---

## 12. MLflow Experiment Tracking

```python
import mlflow
mlflow.set_experiment("xgboost_experiments")

with mlflow.start_run(run_name="xgb_v1"):
    mlflow.log_params({"n_estimators":200, "learning_rate":0.05, "max_depth":4})
    model.fit(X_train, y_train)
    mlflow.log_metric("auc", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
    mlflow.xgboost.log_model(model, "model")

# View all experiments: run `mlflow ui` in terminal → localhost:5000
```

---

## 13. Complete Interview Q&A

**Q1: What is XGBoost and how does it work?**
> XGBoost is a gradient boosting algorithm that builds trees sequentially. Each tree corrects the residual errors of the previous trees. The final prediction is a weighted sum of all trees multiplied by the learning rate.

**Q2: How is XGBoost different from Random Forest?**
> Random Forest builds trees in parallel independently and averages results (bagging). XGBoost builds trees sequentially where each tree corrects previous errors (boosting). XGBoost typically outperforms Random Forest but is more prone to overfitting if not tuned properly.

**Q3: What does learning_rate do?**
> Learning rate controls how much each tree contributes to the final prediction. Lower learning rate means each tree contributes less, requiring more trees (n_estimators) but usually giving better results. Rule of thumb: halve the learning rate and double the n_estimators.

**Q4: How do you prevent overfitting in XGBoost?**
> Multiple strategies: (1) Early stopping — stop adding trees when validation loss stops improving. (2) Lower max_depth — shallower trees generalise better. (3) subsample and colsample_bytree — use random subsets of rows and features. (4) L1/L2 regularisation via reg_alpha and reg_lambda. (5) Lower learning_rate.

**Q5: What is early stopping?**
> Set n_estimators high (e.g., 1000) and early_stopping_rounds to 50. Training stops if validation loss doesn't improve for 50 consecutive rounds. Use model.best_iteration to see how many trees were actually used. Prevents overfitting automatically.

**Q6: What is the difference between weight, gain, and cover feature importance?**
> Weight = how many times a feature appears in splits (biased toward features with many unique values). Gain = average improvement in loss from splits using that feature (best for true importance). Cover = average number of samples impacted by splits on that feature. I always use gain for real analysis.

**Q7: How do you handle imbalanced data with XGBoost?**
> Use scale_pos_weight = negative_count/positive_count. This tells XGBoost to weight the minority class higher during training. Also consider adjusting the prediction threshold from 0.5 to a lower value to catch more positive cases.

**Q8: What is SHAP and why use it with XGBoost?**
> SHAP (SHapley Additive exPlanations) explains individual predictions by showing how much each feature pushed the prediction up or down. Important for production because stakeholders need to understand WHY a model made a specific decision, not just what it predicted.

**Q9: When would you use XGBoost vs LightGBM?**
> On small datasets (<10K rows) XGBoost can be faster. On large datasets (>100K rows) LightGBM is significantly faster due to histogram-based splitting. Accuracy is similar in both cases — usually within 0.5-1%.

**Q10: How do you put XGBoost in production?**
> Wrap it in a sklearn Pipeline with preprocessing steps. Save with joblib. Load in a FastAPI endpoint. The Pipeline ensures preprocessing is never forgotten in production.

---

## 14. Quick Code Template (Use in Any Interview/Project)

```python
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import shap, mlflow, joblib

# 1. Load + prep data
df   = sns.load_dataset('titanic')
df   = df[['pclass','sex','age','fare','survived']].dropna()
df['sex'] = df['sex'].map({'male':0,'female':1})
X, y = df.drop(columns=['survived']), df['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Pipeline
pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale',  StandardScaler()),
    ('model',  XGBClassifier(n_estimators=200, learning_rate=0.05,
                              max_depth=4, random_state=42, eval_metric='logloss'))
])

# 3. Train
pipe.fit(X_train, y_train)

# 4. Evaluate
print(f"AUC: {roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1]):.4f}")
print(f"CV:  {cross_val_score(pipe, X, y, cv=5, scoring='roc_auc').mean():.4f}")

# 5. Explain
model = pipe['model']
explainer = shap.TreeExplainer(model)
shap.summary_plot(explainer.shap_values(X_test), X_test,
                  feature_names=X.columns.tolist())

# 6. Save
joblib.dump(pipe, 'xgboost_pipeline.pkl')
```

---

## 15. What You Have Completed

```
✅ XGBoost Classifier (Titanic — 7 cells)
✅ Early stopping
✅ Feature importance (weight, gain, cover)
✅ SHAP values
✅ MLflow experiment tracking
✅ Learning rate experiments
✅ XGBoost Regressor (California Housing)
✅ XGBoost Multi-class (Iris)
✅ Cross validation
✅ Save + Load model
✅ Imbalanced data (scale_pos_weight)
✅ Pipeline with XGBoost

Mastery level: 95% ✅
Interview ready: Yes ✅
```
