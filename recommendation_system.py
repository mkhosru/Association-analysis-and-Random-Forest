import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
file_path = r'C:\Users\USER\OneDrive\Desktop\PA\Report_of_Khosru_Part_1\fau_clinic_recommender_system.csv'
data = pd.read_csv(file_path)

# Display the first few rows
print("Dataset Preview:")
print(data.head())

# Preprocess the dataset
# Combine relevant features into a single string
data['combined_features'] = data['teams'] + "" + data['previous_experience'] + "" + data['hobbies'] + "" + data['sports']

# Vectorize the combined features
count_vectorizer = TfidfVectorizer().fit_transform(data['combined_features'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(count_vectorizer)

# Create a mapping of Employee_IDs to indices
indices = pd.Series(data.index, index=data['id']).to_dict()

# Define the get_recommendations function
def get_recommendations(employee_id, cosine_sim=cosine_sim):
    # Get the index of the employee that matches the employee ID
    idx = indices[employee_id]
    
    # Get the pairwise similarity scores of all employees with the specified employee
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort employees based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the three most similar employees (excluding the employee itself)
    sim_scores = sim_scores[1:4]
    
    # Get employee indices
    employee_indices = [i[0] for i in sim_scores]
    
    # Return the top three most similar employees
    return data[['id', 'teams', 'hobbies', 'sports']].iloc[employee_indices]

# Get recommendations for emp_050
recommendations = get_recommendations('emp_050', cosine_sim)

# Display recommendations
print("Top 3 Recommendations for emp_050:")
print(recommendations)

# Save recommendations for documentation
recommendations.to_csv("top_3_similar_employees.csv", index=False)
