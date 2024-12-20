# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus

# Step 1: Import the dataset into a DataFrame
file_path = r"C:\Users\USER\OneDrive\Desktop\PA\Report_of_Khosru_Part_1\fau_medical_staff.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Step 2: Data Preprocessing
# Replace 'X' with 1 (indicating active shifts) and NaNs with 0
df = df.fillna(0).map(lambda x: 1 if x == "X" else x)

# Extract the average number of patients needing care
avg_patient_num = df["Avg_Patient_Number"].values

# Extract binary matrix for shifts
shifts = df[["Shift 1", "Shift 2", "Shift 3"]].values

# Wage rates per shift (last row of the dataset)
wage_rates = [45, 45, 60]  # Given wage rates for Shift 1, Shift 2, and Shift 3

# Service level: 1 worker per 4 patients per hour
service_level = 1 / 4

# Step 3: Decision Variables
# Define the decision variables: Number of workers needed for each shift
num_workers = LpVariable.dicts("num_workers", [f"Shift_{j+1}" for j in range(3)], lowBound=0, cat="Integer")

# Step 4: Create the Optimization Problem
# Objective: Minimize the total wage costs
prob = LpProblem("Optimal_Staffing", LpMinimize)

# Add wage cost to the objective function
prob += lpSum([wage_rates[j] * num_workers[f"Shift_{j+1}"] for j in range(3)])

# Step 5: Add Constraints
# Ensure the number of workers meets the service level for all time windows
for t in range(len(avg_patient_num)):  # Loop through time windows
    for j in range(3):  # Loop through shifts
        if shifts[t][j] == 1:  # Active shift
            prob += num_workers[f"Shift_{j+1}"] >= avg_patient_num[t] * service_level

# Step 6: Solve the Problem
prob.solve()

# Step 7: Print Results
print("Status:", LpStatus[prob.status], "Staff Need")
for shift in num_workers:
    print(f"The number of workers needed for {shift} is {int(num_workers[shift].value())} workers.")

# Extract shift names and worker counts
shifts = ['Shift 1', 'Shift 2', 'Shift 3']
workers_needed = [num_workers['Shift_1'].value(), num_workers['Shift_2'].value(), num_workers['Shift_3'].value()]

# Plot the results
plt.bar(shifts, workers_needed, color='skyblue')
plt.xlabel('Shifts')
plt.ylabel('Number of Workers')
plt.title('Optimal Number of Workers per Shift')
plt.tight_layout()
plt.show()
