import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Step 1: Load the dataset
data = pd.read_csv("clinic_performance.csv")

# Step 2: Exploratory Data Analysis (EDA)
print("Dataset Preview:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Drop irrelevant columns
if 'EmpNumber' in data.columns:
    data = data.drop(columns=['EmpNumber'])
    print("\nDropped 'EmpNumber' column.")



# Step 3: Performance Analysis by Job Role
job_role_performance = data.groupby('EmpJobRole')['PerformanceRating'].mean()
job_role_performance_sorted = job_role_performance.sort_values(ascending=False) # Descending order
print("\nAverage Performance Rating by Job Role:")
print(job_role_performance_sorted)  

# Step 4: Data Preprocessing
# Convert categorical columns to numerical using one-hot encoding
data_encoded = pd.get_dummies(data, drop_first=True)
print("\nEncoded Dataset Preview:")
print(data_encoded.head())

# Visualization of Performance Rating by Job Role
plt.figure(figsize=(10, 6))
job_role_performance_sorted.plot(kind='bar', color='skyblue', edgecolor='black')

# Plot formatting
plt.title('Average Performance Rating by Job Role', fontsize=16)
plt.xlabel('Job Role', fontsize=12)
plt.ylabel('Average Performance Rating', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# Step 5: Correlation Analysis
correlation_matrix = data_encoded.corr()

# Step 6: Visualization with Improved Correlation Heatmap
plt.figure(figsize=(18, 15))  # Adjust the figure size
sns.heatmap(
    correlation_matrix, 
    annot=True,                # Display correlation values in the heatmap
    fmt=".2f",                 # Format the values to 2 decimal places
    cmap="coolwarm",           # Use a diverging color map (blue to red)
    cbar_kws={'shrink': 0.8},  # Shrink the color bar for better fit
    linewidths=0.5             # Add space between cells
)
# Correlation of features with 'PerformanceRating' and sorting
important_factors = correlation_matrix['PerformanceRating'].sort_values(ascending=False)[1:11]
print("Top 10 Important Factors (Correlation):")
print(important_factors)


# Customize axes labels
plt.title("Correlation Matrix of Employee Performance Data", fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()



# Define features and target
X = data_encoded.drop(columns=['PerformanceRating'])
y = data_encoded['PerformanceRating']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred, output_dict=True)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(classification_report(y_test, y_pred))



# Step 6: Extract Precision, Recall, and F1-Score for each class
classes = [str(cls) for cls in report.keys() if cls.isdigit()]
precision = [report[cls]['precision'] for cls in classes]
recall = [report[cls]['recall'] for cls in classes]
f1_score = [report[cls]['f1-score'] for cls in classes]

# Step 7: Plot grouped bar chart
x = np.arange(len(classes))  # Label positions
bar_width = 0.25

plt.figure(figsize=(10, 6))

# Plot Precision, Recall, and F1-Score as grouped bars
plt.bar(x - bar_width, precision, width=bar_width, label='Precision', color='tab:blue')
plt.bar(x, recall, width=bar_width, label='Recall', color='tab:orange')
plt.bar(x + bar_width, f1_score, width=bar_width, label='F1-Score', color='tab:green')

# Add values on top of each bar
for i in range(len(classes)):
    plt.text(x[i] - bar_width, precision[i] + 0.02, f"{precision[i]:.2f}", ha='center', fontsize=9)
    plt.text(x[i], recall[i] + 0.02, f"{recall[i]:.2f}", ha='center', fontsize=9)
    plt.text(x[i] + bar_width, f1_score[i] + 0.02, f"{f1_score[i]:.2f}", ha='center', fontsize=9)

# Add overall accuracy as annotation
plt.text(-0.5, 1.02, f"Accuracy: {accuracy:.2f}", fontsize=12, color='black',
         bbox=dict(facecolor='yellow', alpha=0.5))

# Customize the plot
plt.xticks(x, classes, fontsize=10)
plt.xlabel("Classes", fontsize=12)
plt.ylabel("Scores", fontsize=12)
plt.title("Precision, Recall, and F1-Score by Class", fontsize=14)
plt.legend()
plt.ylim(0, 1.1)  # Set y-axis limit
plt.tight_layout()

# Display the plot
plt.show()

# Get feature importance after training the Random Forest model
feature_importance = pd.DataFrame({
'Column_Name': X_train.columns,'Importance Correlation': model.feature_importances_})
feature_importance_sorted = feature_importance.sort_values(by='Importance Correlation', ascending=False)[1:6]
print("\nTop 5 Important Features (Random Forest):")
print(feature_importance_sorted)
