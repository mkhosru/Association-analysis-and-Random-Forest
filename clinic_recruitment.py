import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load the dataset
file_path = "fau_clinic_recruitment.csv" 
data = pd.read_csv(file_path)

# Define the columns related to skills and the target specialization
original_categorical_columns = ["gender", "education"]
skills_columns = [
    "communication", "empathy", "confidence", "professional", "patience",
    "critical thinking", "support", "management", "commitment"
]
specialization_column = "critical_care_nursing"

# Create experience ranges
data['experience_range'] = pd.cut(data['experience'], bins=[0, 8, 21, float('inf')], labels=["[0,8)", "[8,20]", "[20,inf)"], right=False)

# Convert categorical columns to binary format using one-hot encoding
data_encoded = pd.get_dummies(data, columns=original_categorical_columns + ['experience_range'], drop_first=True)

# Update skills_columns to include the new one-hot encoded column names
skills_columns += [col for col in data_encoded.columns if col.startswith(tuple(original_categorical_columns)) or col.startswith("experience_range")]

# Prepare a binary dataset for skills and specialization
binary_data = data_encoded[skills_columns + [specialization_column]]

# Ensure all values are boolean
binary_data = binary_data > 0  # Convert to boolean

# Generate frequent itemsets using the apriori algorithm
frequent_itemsets = apriori(binary_data, min_support=0.02, use_colnames=True)

# Add the required 'num_itemsets' argument for older mlxtend versions
frequent_itemsets['num_itemsets'] = len(frequent_itemsets)

# Generate association rules
rules = association_rules(
    frequent_itemsets, 
    num_itemsets=len(frequent_itemsets),  # Required for older mlxtend versions
    metric="confidence", 
    min_threshold=0.25
)

# Filter rules for the critical care nursing specialization and single consequents
critical_care_rules = rules[
    (rules['consequents'].apply(lambda x: specialization_column in x)) & 
    (rules['consequents'].apply(lambda x: len(x) == 1))
]

# Sort rules to prioritize `gender=m` in the antecedents
critical_care_rules['lhs'] = critical_care_rules['antecedents'].apply(lambda x: frozenset(sorted(x, key=lambda item: item != 'gender_m')))

# Add the 'coverage' column for rules
critical_care_rules['coverage'] = critical_care_rules['antecedent support']

# Add a count column
critical_care_rules['count'] = critical_care_rules['support'] * len(binary_data)

# Sort the rules by lift, support, and confidence
sorted_rules = critical_care_rules.sort_values(by=['lift', 'support', 'confidence'], ascending=[False, False, False])

# Rename columns to match the output in the screenshot
sorted_rules.rename(columns={
    'lhs': 'lhs',
    'consequents': 'rhs',
}, inplace=True)

# Reset the index to start from 1
sorted_rules.reset_index(drop=True, inplace=True)
sorted_rules.index += 1

# Select and display the relevant columns
output_columns = ['lhs', 'rhs', 'support', 'confidence', 'coverage', 'lift', 'count']
formatted_rules = sorted_rules[output_columns]

pd.set_option('display.max_colwidth', None)  # Allows displaying the full width of columns
pd.set_option('display.width', 1000)

print("Top 10 Rules for Critical Care Nursing:")
print(formatted_rules.head(10))
