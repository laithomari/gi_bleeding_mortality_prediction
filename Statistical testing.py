# Importing libraries
import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import chi2_contingency, fisher_exact, ttest_ind, mannwhitneyu, levene, kstest
import statsmodels.stats.api as sms
from statsmodels.stats.contingency_tables import mcnemar

# Importing data
df = pd.read_csv('your_path.csv')

# Separate categorical and continuous columns
categorical_cols = ['gender', 'ams', 'hematemesis', 'melena', 'malignancy', 'metastatic_malignancy', 'liver_disease', 'acute_mi', 'chf']
continuous_cols = [col for col in df.columns if col not in categorical_cols + ['target']]

# Tables to store results
categorical_results = []
continuous_results = []

# Function to calculate mean ± 95% CI
def mean_ci(data):
    mean = np.mean(data)
    ci = sms.DescrStatsW(data).tconfint_mean()
    return f"{mean:.2f} ± ({ci[0]:.2f}, {ci[1]:.2f})"

# Test for normality with Kolmogorov-Smirnov test for large datasets
def test_normality(data):
    if len(data) < 3:
        return False  # Insufficient data for normality test
    if len(data) > 5000:
        stat, p = kstest(data, 'norm', args=(np.mean(data), np.std(data)))
    else:
        stat, p = shapiro(data)
    return p >= 0.05  # True if data is normally distributed


# Analyzing Categorical Variables
for col in categorical_cols:
    # Create a cross-tabulation for the variable by target
    crosstab = pd.crosstab(df['target'], df[col])

    # Calculate counts and percentages for each group
    count_0 = crosstab.loc[0, 1] if 1 in crosstab.columns else 0
    count_1 = crosstab.loc[1, 1] if 1 in crosstab.columns else 0
    total_0 = crosstab.loc[0].sum()
    total_1 = crosstab.loc[1].sum()

    perc_0 = (count_0 / total_0 * 100).round(2) if total_0 > 0 else 0
    perc_1 = (count_1 / total_1 * 100).round(2) if total_1 > 0 else 0

    # Create contingency table for statistical test
    contingency_table = [
        [count_0, total_0 - count_0],
        [count_1, total_1 - count_1]
    ]

    # Perform Chi-square or Fisher's Exact Test
    if contingency_table[0][0] + contingency_table[1][0] > 0:
        if all(sum(row) > 0 for row in contingency_table):  # Ensure no zero rows for chi-square
            _, p = chi2_contingency(contingency_table)[:2]
        else:
            _, p = fisher_exact(contingency_table)
    else:
        p = np.nan  # Handle cases where there's no data

    result = {
        "Variable": col,
        "Target 0 Count (%)": f"{count_0} ({perc_0}%)",
        "Target 1 Count (%)": f"{count_1} ({perc_1}%)",
        "p-value": "<0.0001" if p < 0.0001 else f"{p:.4f}" if not np.isnan(p) else "N/A"
    }
    categorical_results.append(result)



# Analyzing Continuous Variables
for col in continuous_cols:
    # Split data by target groups
    group_0 = df[df['target'] == 0][col].dropna()
    group_1 = df[df['target'] == 1][col].dropna()

    # Skip variable if one of the groups has less than 3 samples
    if len(group_0) < 3 or len(group_1) < 3:
        result = {
            "Variable": col,
            "Group 0 Mean ± 95% CI": "Insufficient data",
            "Group 1 Mean ± 95% CI": "Insufficient data",
            "p-value": "Insufficient data"
        }
        continuous_results.append(result)
        continue

    # Test for normality
    norm_0 = test_normality(group_0)
    norm_1 = test_normality(group_1)

    # Test for homogeneity of variances
    homogeneity = levene(group_0, group_1).pvalue >= 0.05

    # Select appropriate statistical test
    if norm_0 and norm_1 and homogeneity:
        stat, p = ttest_ind(group_0, group_1)
    else:
        stat, p = mannwhitneyu(group_0, group_1)

    # Calculate mean ± 95% CI for each group
    ci_0 = mean_ci(group_0)
    ci_1 = mean_ci(group_1)

    result = {
        "Variable": col,
        "Group 0 Mean ± 95% CI": ci_0,
        "Group 1 Mean ± 95% CI": ci_1,
        "p-value": "<0.0001" if p < 0.0001 else f"{p:.4f}"
    }
    continuous_results.append(result)

# Convert results to DataFrames for easier display and export
categorical_df = pd.DataFrame(categorical_results)
continuous_df = pd.DataFrame(continuous_results)


# Print results to the console
print("Categorical Analysis Results:")
print(categorical_df)
print("\nContinuous Analysis Results:")
print(continuous_df)

# DeLong test to compare AUCs
def Delong_test(true, prob_A, prob_B):

    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=np.float64)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1)
            i = j
        T2 = np.empty(N, dtype=np.float64)
        T2[J] = T + 1
        return T2

    def compute_ground_truth_statistics(true):
        assert np.array_equal(np.unique(true), [0, 1]), 
        order = (-true).argsort()
        label_1_count = int(true.sum())
        return order, label_1_count

    # Prepare data
    order, label_1_count = compute_ground_truth_statistics(np.array(true))
    sorted_probs = np.vstack((np.array(prob_A), np.array(prob_B)))[:, order]

    # Fast DeLong computation starts here
    m = label_1_count  # Number of positive samples
    n = sorted_probs.shape[1] - m  # Number of negative samples
    k = sorted_probs.shape[0]  # Number of models (2)

    # Initialize arrays for midrank computations
    tx, ty, tz = [np.empty([k, size], dtype=np.float64) for size in [m, n, m + n]]
    for r in range(k):
        positive_examples = sorted_probs[r, :m]
        negative_examples = sorted_probs[r, m:]
        tx[r, :], ty[r, :], tz[r, :] = [
            compute_midrank(examples) for examples in [positive_examples, negative_examples, sorted_probs[r, :]]
        ]

    # Calculate AUCs
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)

    # Compute variance components
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

    # Compute covariance matrices
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n

    # Calculating z-score and p-value
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, delongcov), l.T)).flatten()
    p_value = scipy.stats.norm.sf(abs(z)) * 2

    z_score = -z[0].item()
    p_value = p_value[0].item()

    return z_score, p_value

# Comparing AUC of the model vs AIMS65
z_score, p_value = Delong_test(y_test, y_proba_all, aims65_test)
print(f"Z-Score: {z_score}, P-Value: {p_value}")

# Comparing AUC of the model vs GBS
z_score, p_value = Delong_test(y_test, y_proba_all, gbs_test)
print(f"Z-Score: {z_score}, P-Value: {p_value}")



# Replace these values with your actual counts from the test set:
ml_vs_aims65_contingency = np.array([
    [50, 3],  
    [1, 0]    
])

ml_vs_gbs_contingency = np.array([
    [59, 0],  
    [1, 0]    
])

# Perform McNemar test for our model vs AIMS65
result_aims65 = mcnemar(ml_vs_aims65_contingency, exact=False, correction=True)
print(f"ML vs AIMS65: χ² = {result_aims65.statistic:.2f}, p = {result_aims65.pvalue:.4f}")

# Perform McNemar test for our model vs GBS
result_gbs = mcnemar(ml_vs_gbs_contingency, exact=False, correction=True)
print(f"ML vs GBS: χ² = {result_gbs.statistic:.2f}, p = {result_gbs.pvalue:.4f}")