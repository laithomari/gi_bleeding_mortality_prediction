# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve
)
from scipy.stats import randint, uniform, sem, t
from sklearn.base import BaseEstimator, ClassifierMixin
import shap

############################################################################################
############################################################################################

# Importing data and initial preprocessing
df = pd.read_csv('/data_path.csv')

#Imputing gender
gender_mapper = {'M' : 1, 'F' : 0}
df['gender'] = df['gender'].map(gender_mapper)

# Calculate the percentage of missing values in each column
missing_percentage = df.isnull().sum() * 100 / len(df)
# Drop columns with more than 50% missing values
columns_to_drop = missing_percentage[missing_percentage > 50].index

# Keep a copy for the AIMS and GBS calculations
aims_df = df.copy()
gbs_df = df.copy()

dfs = [df, aims_df, gbs_df]
# Fill missing values with the mean of each column
for dataset in dfs:
  for col in dataset.columns:
      if dataset[col].isnull().any():
          dataset[col] = dataset[col].fillna(dataset[col].mean())

# Data partitioning
X, y = df.drop('target', axis = 1), df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42) 

# select or drop specific columns
# Drop the specified columns
columns_to_drop = ['alp', 'ast', 'alt', 'tbili', 'o2sat', 'resprate', 'glucose', 'ptt', 'temperature', 'gender', 'acute_mi', 'chf', 'hematocrit', 'dbp', 'sodium', 'potassium',
                   'ams', 'metastatic_malignancy', 'pt', 'chloride', 'anion_gap']
X_train = X_train.drop(columns=columns_to_drop, errors='ignore')
X_test = X_test.drop(columns=columns_to_drop, errors='ignore')

# K-means clustering, model training, fine-tuning, and evaluation
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform

###############################################################################
# 1. TRAINING PHASE
###############################################################################


majority_class = 0
minority_class = 1

X_major = X_train[y_train == majority_class]
y_major = y_train[y_train == majority_class]

X_minor = X_train[y_train == minority_class]
y_minor = y_train[y_train == minority_class]

L = 24  # number of clusters
kmeans = KMeans(n_clusters=L, random_state=42)
kmeans.fit(X_major)
clusters = kmeans.predict(X_major)

param_distributions = {
    'n_estimators': [50, 100, 200, 300, 400],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 10),
    'bootstrap': [True, False],
    'class_weight': [None, 'balanced', 'balanced_subsample']
}

models = []

for cluster_id in range(L):
    X_cluster = X_major[clusters == cluster_id]
    y_cluster = y_major[clusters == cluster_id]

    X_train_cluster = np.vstack((X_cluster, X_minor))
    y_train_cluster = np.concatenate((y_cluster, y_minor), axis=0)

    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=20,
        scoring='roc_auc',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train_cluster, y_train_cluster)
    best_rf = random_search.best_estimator_
    print(f"Cluster {cluster_id} best params: {random_search.best_params_}")
    models.append(best_rf)

###############################################################################
# 2. 5-FOLD CROSS-VALIDATION WITH REORDERING
###############################################################################

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
THRESHOLD = 0.852

# Initialize storage for indices and predictions
test_indices = []
y_test_parts = []
y_pred_parts = []
y_proba_parts = []

# Metrics storage
auc_scores = []
acc_scores = []
prec_scores = []
rec_scores = []
spec_scores = []
npv_scores = []

for train_idx, test_idx in kfold.split(X_test, y_test):
    X_test_fold = X_test.iloc[test_idx]
    y_test_fold = y_test.iloc[test_idx]

    # Get predictions from all models
    probas_list = [m.predict_proba(X_test_fold)[:, 1] for m in models]
    avg_proba_fold = np.mean(probas_list, axis=0)
    y_pred_fold = (avg_proba_fold >= THRESHOLD).astype(int)

    # Store indices and predictions for reordering
    test_indices.append(test_idx)
    y_test_parts.append(y_test_fold)
    y_pred_parts.append(y_pred_fold)
    y_proba_parts.append(avg_proba_fold)

    # Calculate fold metrics
    fold_auc = roc_auc_score(y_test_fold, avg_proba_fold)
    fold_acc = accuracy_score(y_test_fold, y_pred_fold)
    fold_prec = precision_score(y_test_fold, y_pred_fold, zero_division=0)
    fold_rec = recall_score(y_test_fold, y_pred_fold, zero_division=0)

    cm = confusion_matrix(y_test_fold, y_pred_fold)
    tn, fp, fn, tp = cm.ravel()
    fold_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    fold_npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    # Store metrics
    auc_scores.append(fold_auc)
    acc_scores.append(fold_acc)
    prec_scores.append(fold_prec)
    rec_scores.append(fold_rec)
    spec_scores.append(fold_spec)
    npv_scores.append(fold_npv)

# Reconstruct in original order
test_indices = np.concatenate(test_indices)
sort_order = np.argsort(test_indices)

y_test_all = np.concatenate(y_test_parts)[sort_order]
y_pred_all = np.concatenate(y_pred_parts)[sort_order]
y_proba_all = np.concatenate(y_proba_parts)[sort_order]

###############################################################################
# 3. METRICS & REPORTING
###############################################################################

def mean_ci(values, alpha=0.95):
    arr = np.array(values)
    mean_val = np.mean(arr)
    std_val = np.std(arr, ddof=1)
    n = len(arr)
    z = 1.96
    margin = z * (std_val / np.sqrt(n))
    return mean_val, (mean_val - margin, mean_val + margin)

# Calculate metrics with CI
auc_mean, auc_ci = mean_ci(auc_scores)
acc_mean, acc_ci = mean_ci(acc_scores)
prec_mean, prec_ci = mean_ci(prec_scores)
rec_mean, rec_ci = mean_ci(rec_scores)
spec_mean, spec_ci = mean_ci(spec_scores)
npv_mean, npv_ci = mean_ci(npv_scores)

print("\n=== 5-Fold CV Metrics (Mean ± 95% CI) ===")
print(f"AUC:        {auc_mean:.4f}  ({auc_ci[0]:.4f}-{auc_ci[1]:.4f})")
print(f"Accuracy:   {acc_mean:.4f}  ({acc_ci[0]:.4f}-{acc_ci[1]:.4f})")
print(f"Precision:  {prec_mean:.4f} ({prec_ci[0]:.4f}-{prec_ci[1]:.4f})")
print(f"Recall:     {rec_mean:.4f}  ({rec_ci[0]:.4f}-{rec_ci[1]:.4f})")
print(f"Specificity:{spec_mean:.4f} ({spec_ci[0]:.4f}-{spec_ci[1]:.4f})")
print(f"NPV:        {npv_mean:.4f}  ({npv_ci[0]:.4f}-{npv_ci[1]:.4f})")

# Final confusion matrix and classification report
cm_agg = confusion_matrix(y_test_all, y_pred_all)
cm_df = pd.DataFrame(
    cm_agg,
    index=['Actual Negative', 'Actual Positive'],
    columns=['Predicted Negative', 'Predicted Positive']
)

print("\n=== Aggregated Confusion Matrix ===")
print(cm_df)
print("\n=== Classification Report ===")
print(classification_report(y_test_all, y_pred_all))


# Identifying ideal threshold using Youden J
# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test_all, y_proba_all)

# Calculate specificity
specificity = 1 - fpr

# Calculate Youden's J statistic for each threshold
youden_j = tpr + specificity - 1

# Find the threshold that maximizes Youden's J
optimal_idx = np.argmax(youden_j)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal Threshold = {optimal_threshold:.3f}")

#SHAP analysis for interpretability
###############################################################################
# 1. CREATE AN ENSEMBLE WRAPPER
###############################################################################
# This wrapper acts as a single "black-box" model to the SHAP explainer.

class EnsembleWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, models):
        self.models = models
        self._estimator_type = "classifier"

    def fit(self, X, y):
        """No re-training needed; just for scikit-learn compliance."""
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        """Average probabilities from all RF models."""
        probas = [m.predict_proba(X) for m in self.models]
        return np.mean(probas, axis=0)  # shape (n_samples, 2)

    def predict(self, X):
        """Threshold at 0.5 on the ensemble's average positive-class probability."""
        avg_proba = self.predict_proba(X)
        return (avg_proba[:, 1] >= 0.5).astype(int)

###############################################################################
# 2. CREATE A PREDICTION FUNCTION FOR CLASS=1 PROBABILITY
###############################################################################
# We'll pass this function to KernelExplainer, which expects f(x) -> scalar output
# (here, the probability of the positive class).

def ensemble_positive_proba(X):
    """
    Return p(class=1) for each row in X,
    using the ensemble's average probabilities.
    """
    # Make sure X is DataFrame if your ensemble expects that.
    # If your wrapper accepts NumPy arrays, no conversion is needed.
    # We'll assume X is a NumPy array or DataFrame that your models can handle.
    # The ensemble wrapper needs a DataFrame if your models expect it.
    # If your models are fine with arrays, no conversion is required.
    if isinstance(X, pd.DataFrame):
        return ensemble_estimator.predict_proba(X)[:, 1]
    else:
        return ensemble_estimator.predict_proba(X)[:, 1]

###############################################################################
# 3. FIT THE ENSEMBLE WRAPPER ONCE AND EVALUATE
###############################################################################

# If X_test is a NumPy array, convert to DataFrame if needed for consistent usage:
if not isinstance(X_test, pd.DataFrame):
    X_test = pd.DataFrame(X_test, columns=[f"Feature_{i}" for i in range(X_test.shape[1])])

ensemble_estimator = EnsembleWrapper(models)
ensemble_estimator.fit(X_test, y_test)  # just sets self.classes_

baseline_probs = ensemble_estimator.predict_proba(X_test)[:, 1]
baseline_auc = roc_auc_score(y_test, baseline_probs)
print(f"Baseline Ensemble AUC on Test Set: {baseline_auc:.4f}")

###############################################################################
# 4. BUILD A KERNEL EXPLAINER FOR SHAP
###############################################################################
# Treating the ensemble as a black-box function for class=1 probability.

# 4a. CREATE A BACKGROUND SAMPLE
#    KernelExplainer uses a "background" to estimate the distribution of features.
#    Taking a small random sample of X_train or X_test

BACKGROUND_SIZE = 100  # adjust as needed
if isinstance(X_train, pd.DataFrame):
    background_df = X_train.sample(min(BACKGROUND_SIZE, len(X_train)), random_state=42)
else:
    # If X_train is an array
    idx = np.random.choice(len(X_train), size=min(BACKGROUND_SIZE, len(X_train)), replace=False)
    background_df = pd.DataFrame(X_train[idx], columns=[f"Feature_{i}" for i in range(X_train.shape[1])])

# 4b. Initialize KernelExplainer with ensemble's class-1 probability function
explainer = shap.KernelExplainer(
    ensemble_positive_proba,
    background_df,  # background sample
)

###############################################################################
# 5. COMPUTE SHAP VALUES FOR THE TEST SET
###############################################################################
# For large X_test, consider only a subset for interpretability to reduce runtime.

TEST_SUBSET_SIZE = 200
if len(X_test) > TEST_SUBSET_SIZE:
    X_test_subset = X_test.sample(TEST_SUBSET_SIZE, random_state=42)
else:
    X_test_subset = X_test

# shap_values => array of shape (n_samples, n_features)
# KernelExplainer returns a single array for "f(x)->scalar" usage
shap_values = explainer.shap_values(X_test_subset, nsamples=100)
# nsamples=100 is how many times KernelExplainer samples feature perturbations
# (higher => better accuracy but slower)

###############################################################################
# 6. VISUALIZE THE RESULTS
###############################################################################
# We'll use shap's summary_plot to see global feature importance
# (top features with largest average absolute SHAP).

shap.initjs()
shap.summary_plot(shap_values, X_test_subset, plot_type='dot')
plt.tight_layout()
plt.show()

# Individual force plots or waterfall plots
# for local interpretability on a single instance:
sample_idx = 0  # pick a row in X_test_subset
shap_values_single = shap_values[sample_idx]

shap.force_plot(
    explainer.expected_value,
    shap_values_single,
    X_test_subset.iloc[sample_idx],
    link="identity"
)

#AIMS65 calculation and evaluation
# Preprocess
X_aims, y_aims = aims_df.drop('target', axis = 1), aims_df['target']
X_train_aims, X_test_aims, y_train_aims, y_test_aims = train_test_split(X_aims, y_aims, test_size = 0.2, stratify = y_aims, random_state = 42)

X_gbs, y_gbs = gbs_df.drop('target', axis = 1), gbs_df['target']
X_train_gbs, X_test_gbs, y_train_gbs, y_test_gbs = train_test_split(X_gbs, y_gbs, test_size = 0.2, stratify = y_gbs, random_state = 42)

# 1. Define a function to calculate AIMS65
def calculate_aims65(df):

    aims65 = (
        (df['age'] >= 65).astype(int) +         # Age ≥ 65
        (df['albumin'] < 3).astype(int) +       # Albumin < 3
        (df['inr'] > 1.5).astype(int) +         # INR > 1.5
        (df['sbp'] <= 90).astype(int) +         # SBP ≤ 90
        (df['ams'] == 1).astype(int)            # AMS present
    )
    return aims65

# 2. Define a function to calculate additional metrics
def calculate_metrics(y_true, scores, threshold=1):

    # Binarize scores based on the threshold
    y_pred = (scores >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    auc = roc_auc_score(y_true, scores)

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'auc': auc,
        'confusion_matrix': (tn, fp, fn, tp)
    }

# 3. Calculate AIMS65 scores for training and test sets
aims65_train = calculate_aims65(X_train_aims)
aims65_test = calculate_aims65(X_test_aims)

# 4. Calculate metrics for training and test sets
threshold = 1  
train_metrics = calculate_metrics(y_train_aims, aims65_train, threshold=threshold)
test_metrics = calculate_metrics(y_test_aims, aims65_test, threshold=threshold)

# 5. Display results
print("\n=== AIMS65 Performance on Training Set ===")
print(f"Sensitivity: {train_metrics['sensitivity']:.4f}")
print(f"Specificity: {train_metrics['specificity']:.4f}")
print(f"PPV:         {train_metrics['ppv']:.4f}")
print(f"NPV:         {train_metrics['npv']:.4f}")
print(f"AUC:         {train_metrics['auc']:.4f}")
print(f"Confusion Matrix: TN={train_metrics['confusion_matrix'][0]}, "
      f"FP={train_metrics['confusion_matrix'][1]}, FN={train_metrics['confusion_matrix'][2]}, "
      f"TP={train_metrics['confusion_matrix'][3]}")

print("\n=== AIMS65 Performance on Test Set ===")
print(f"Sensitivity: {test_metrics['sensitivity']:.4f}")
print(f"Specificity: {test_metrics['specificity']:.4f}")
print(f"PPV:         {test_metrics['ppv']:.4f}")
print(f"NPV:         {test_metrics['npv']:.4f}")
print(f"AUC:         {test_metrics['auc']:.4f}")
print(f"Confusion Matrix: TN={test_metrics['confusion_matrix'][0]}, "
      f"FP={test_metrics['confusion_matrix'][1]}, FN={test_metrics['confusion_matrix'][2]}, "
      f"TP={test_metrics['confusion_matrix'][3]}")

#Glasgow-Blatchford Score (GBS) calculation and evaluation
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

# 1. Define a function to calculate the Glasgow-Blatchford Score (GBS)
def calculate_gbs(df):

    # BUN (mg/dL)
    bun_score = (
        (df['bun'] < 18.2).astype(int) * 0 +
        ((df['bun'] >= 18.2) & (df['bun'] <= 22.3)).astype(int) * 2 +
        ((df['bun'] > 22.3) & (df['bun'] <= 28)).astype(int) * 3 +
        ((df['bun'] > 28) & (df['bun'] <= 70)).astype(int) * 4 +
        (df['bun'] > 70).astype(int) * 6
    )

    # Hemoglobin (g/dL) with encoded gender (1 = male, 0 = female)
    hb_score = (
        (df['gender'] == 1).astype(int) * (
            (df['hemoglobin'] > 13).astype(int) * 0 +
            ((df['hemoglobin'] >= 12) & (df['hemoglobin'] <= 13)).astype(int) * 1 +
            ((df['hemoglobin'] >= 10) & (df['hemoglobin'] < 12)).astype(int) * 3 +
            (df['hemoglobin'] < 10).astype(int) * 6
        ) +
        (df['gender'] == 0).astype(int) * (
            (df['hemoglobin'] > 12).astype(int) * 0 +
            ((df['hemoglobin'] >= 10) & (df['hemoglobin'] <= 12)).astype(int) * 1 +
            (df['hemoglobin'] < 10).astype(int) * 6
        )
    )

    # Systolic blood pressure (SBP)
    sbp_score = (
        (df['sbp'] >= 110).astype(int) * 0 +
        ((df['sbp'] >= 100) & (df['sbp'] < 110)).astype(int) * 1 +
        ((df['sbp'] >= 90) & (df['sbp'] < 100)).astype(int) * 2 +
        (df['sbp'] < 90).astype(int) * 3
    )

    # Other criteria
    other_score = (
        (df['heartrate'] >= 100).astype(int) * 1 +  # Pulse ≥ 100
        (df['melena'] == 1).astype(int) * 1 +      # Melena present
        (df['ams'] == 1).astype(int) * 2 +         # Syncope present
        (df['liver_disease'] == 1).astype(int) * 2 +  # Liver disease history
        (df['chf'] == 1).astype(int) * 2           # Cardiac failure present
    )

    # Total GBS score
    return bun_score + hb_score + sbp_score + other_score

# 2. Define a function to calculate additional metrics
def calculate_metrics(y_true, scores, threshold=1):

    # Binarize scores based on the threshold
    y_pred = (scores >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    auc = roc_auc_score(y_true, scores)

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'auc': auc,
        'confusion_matrix': (tn, fp, fn, tp)
    }

# 3. Calculate Glasgow-Blatchford Scores for training and test sets
gbs_train = calculate_gbs(X_train_gbs)
gbs_test = calculate_gbs(X_test_gbs)

# 4. Calculate metrics for training and test sets
threshold = 2  # Default threshold for GBS
train_metrics = calculate_metrics(y_train_gbs, gbs_train, threshold=threshold)
test_metrics = calculate_metrics(y_test_gbs, gbs_test, threshold=threshold)

# 5. Display results
print("\n=== GBS Performance on Training Set ===")
print(f"Sensitivity: {train_metrics['sensitivity']:.4f}")
print(f"Specificity: {train_metrics['specificity']:.4f}")
print(f"PPV:         {train_metrics['ppv']:.4f}")
print(f"NPV:         {train_metrics['npv']:.4f}")
print(f"AUC:         {train_metrics['auc']:.4f}")
print(f"Confusion Matrix: TN={train_metrics['confusion_matrix'][0]}, "
      f"FP={train_metrics['confusion_matrix'][1]}, FN={train_metrics['confusion_matrix'][2]}, "
      f"TP={train_metrics['confusion_matrix'][3]}")

print("\n=== GBS Performance on Test Set ===")
print(f"Sensitivity: {test_metrics['sensitivity']:.4f}")
print(f"Specificity: {test_metrics['specificity']:.4f}")
print(f"PPV:         {test_metrics['ppv']:.4f}")
print(f"NPV:         {test_metrics['npv']:.4f}")
print(f"AUC:         {test_metrics['auc']:.4f}")
print(f"Confusion Matrix: TN={test_metrics['confusion_matrix'][0]}, "
      f"FP={test_metrics['confusion_matrix'][1]}, FN={test_metrics['confusion_matrix'][2]}, "
      f"TP={test_metrics['confusion_matrix'][3]}")
