# Bradley Ferraro
# Machine Learning CSC 480 Final Project
# Emotions_EEG_MS_Project
# Description: 
#
#
#


from warnings import filterwarnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import csv




df = pd.read_csv("emotions.csv")
# print(df.info())

# Path to your CSV file
file_path = 'emotions.csv'

# ----------------------------------------
# Preprocessing Data
# ----------------------------------------

# checking for missing data or outliers

# Check for missing values
print("Missing Values:\n", df.isnull().sum())
missing_values = df.isna().sum()
total_missing_values = missing_values.sum()
print( "num of missing values: " + str(total_missing_values))


# check for duplicates
duplicates = df.duplicated().sum()
print("Number of duplicate rows:", duplicates)

# handling outliers - using Z-score for columns
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
outliers = (z_scores > 10).any(axis=1)
print("Number of outlier rows:", np.sum(outliers))

# I could remove outliers if needed
# data_cleaned = data[~outliers]
# I am not removing outliers now because I need to scal the 
# data. After scaling and low-variant feature elimination,
# I will check again for outliers



# ----------------------------------------
# Parameter Inspection
# ----------------------------------------

# Inspecting the data present, deciding what is useful

# Determine what information is present in the dataset

# Initialize an empty set for unique sections
unique_sections = set()

# Open the CSV file and read the first row
with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    first_row = next(reader)

    # Process each label in the first row
    for label in first_row:
        section = str(label).split('_')[0]
        unique_sections.add(section)

# Convert the set to a list and print it
# unique_sections_list = list(unique_sections)
# print(unique_sections_list)
# for item in unique_sections:
#     print(item)
#     print()

# This section I am working on doing these steps in the 
# correct order. To my understading I should scale first,
# perform variance-based feature selection next, and then
# encode my labels. 
# ---------------------------------------------------------------

# correlation analysis and variance/distribution analysis

scaler = StandardScaler()
df_2 = df.drop(["label"], axis=1)
variances = df_2.var()

# I think the dataset is too large, these aren't working
# # Compute the correlation matrix
# corr_matrix = np.corrcoef(df_2)
# corr_matrix = df.corr()

# # Visualize the correlation matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
# plt.title('Feature Correlation Matrix')
# plt.show()

# Variance of each feature
# this prints every column's graph (over 2000 graphs)
# print("Variance of each feature:\n", variances)

# # Plotting distributions of each feature
# for column in df_2.columns:
#     plt.figure(figsize=(8, 4))
#     sns.histplot(df[column], kde=True)
#     plt.title(f'Distribution of {column}')
#     plt.show()

# Features are ordered by their variance, 
# so you can easily see a curve that might sharply drop at some point 
# Sort variances in descending order
sorted_variances = variances.sort_values(ascending=False)

# # Plot
# plt.figure(figsize=(15, 5))
# plt.plot(sorted_variances.values)
# plt.ylabel('Variance')
# plt.xlabel('Features')
# plt.title('Feature Variances')
# plt.show()


# the graph has a large spike near zero and the rest have vary little varience.
# Is this because they are numbers based near zero so they have little varience
# relative the the larger numbers? 
# No, because varience takes into account relative size.
# Upon further reading, this is not true, I need to scale it first, 
# then look for low-variance features.
# scaling code





# Scaling
X_scaled = pd.DataFrame(scaler.fit_transform(df_2), columns=df_2.columns)


# Here is code to prune low-varient data as get a usable 
# dataset size for correlation metrics.

# Set a reasonable variance threshold
threshold = 1.0  # test value, adjust based scaling

# Create a VarianceThreshold feature selector
sel = VarianceThreshold(threshold)

# Fit the selector to your scaled data
sel.fit(X_scaled)

# Transform (reduce) your dataset
reduced_df = pd.DataFrame(sel.transform(X_scaled), 
                          columns=X_scaled.columns[sel.get_support()])



# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

print(f"Original number of features: {df_2.shape[1]}")
print(f"Reduced number of features: {reduced_df.shape[1]}")


# checking for outliers again, after scaling and pruning features
# handling outliers - using Z-score for columns
z_scores = np.abs(stats.zscore(reduced_df.select_dtypes(include=[np.number])))
outliers = (z_scores > 10).any(axis=1)
print("Number of outlier rows:", np.sum(outliers))
print(f"Number of features: {reduced_df.shape[0]}")
reduced_df_2 = reduced_df[~outliers]
print(f"Reduced number of features: {reduced_df_2.shape[0]}")

# now that outliers are gone and features or pruned, 
# I can try the correlation matrix again

# corr_matrix = reduced_df.corr()
# # Visualize the correlation matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
# plt.title('Feature Correlation Matrix')
# plt.show()

# still too big for this to run in a reasonable timeframe




# Here I will begin the campare and contrast section of the project.
# I am comparing pca to ica to the larger reduced_df_2 in order to 
# determine which one has the best performance with a couple different
# models. 

# Standardize the data
scaled_data = scaler.fit_transform(reduced_df_2)

# Applying PCA
pca = PCA(n_components=0.95)  # retains 95% of the variance
pca_data = pca.fit_transform(scaled_data)

print(f"Original number of features: {scaled_data.shape[1]}")
print(f"Reduced number of features after PCA: {pca_data.shape[1]}")

# Applying ICA
ica = FastICA(n_components=95, random_state=0)  # Adjust n_components based on your requirement
ica_data = ica.fit_transform(scaled_data)

print(f"Number of independent components: {ica_data.shape[1]}")



# # ----------------------------------------
# # Model Training/Testing and comparison
# # ----------------------------------------

# Function to train and evaluate a model
def train_and_evaluate(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return accuracy

# Assuming reduced_df_2, pca_data, and ica_data have the same number of rows

# Update labels to match the number of samples in the reduced datasets
y_reduced = df['label'][~outliers]  # 'outliers' is a boolean mask used previously

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_reduced)

# Train-test split for each dataset
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(pca_data, y_encoded, test_size=0.2, random_state=42)
X_train_ica, X_test_ica, y_train_ica, y_test_ica = train_test_split(ica_data, y_encoded, test_size=0.2, random_state=42)
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(reduced_df_2, y_encoded, test_size=0.2, random_state=42)

# Initialize models
rf_model = RandomForestClassifier(random_state=42)
svc_model = SVC(random_state=42)


# ------------------------------------------------------------------------------------------------
# Random Forest with PCA data
rf_pca_accuracy = train_and_evaluate(X_train_pca, X_test_pca, y_train_pca, y_test_pca, rf_model)
print("Random Forest with PCA data Accuracy:", rf_pca_accuracy)

# SVC with PCA data
svc_pca_accuracy = train_and_evaluate(X_train_pca, X_test_pca, y_train_pca, y_test_pca, svc_model)
print("SVC with PCA data Accuracy:", svc_pca_accuracy)

# Random Forest with ICA data
rf_ica_accuracy = train_and_evaluate(X_train_ica, X_test_ica, y_train_ica, y_test_ica, rf_model)
print("Random Forest with ICA data Accuracy:", rf_ica_accuracy)

# SVC with ICA data
svc_ica_accuracy = train_and_evaluate(X_train_ica, X_test_ica, y_train_ica, y_test_ica, svc_model)
print("SVC with ICA data Accuracy:", svc_ica_accuracy)

# Random Forest with Reduced data
rf_reduced_accuracy = train_and_evaluate(X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced, rf_model)
print("Random Forest with Reduced data Accuracy:", rf_reduced_accuracy)

# SVC with Reduced data
svc_reduced_accuracy = train_and_evaluate(X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced, svc_model)
print("SVC with Reduced data Accuracy:", svc_reduced_accuracy)

