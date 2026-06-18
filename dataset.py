import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)

# ==========================================
# 1. SETUP & BASE DATA CONFIGURATION
# ==========================================
num_employees = 650  # Between 500-800
start_date = datetime(2023, 1, 1)
end_date = datetime(2025, 12, 31)
date_range = pd.date_range(start_date, end_date)

departments = ['Operations', 'Engineering', 'Customer Support', 'Sales', 'Logistics']
job_roles = {
    'Operations': ['Supervisor', 'Associate'],
    'Engineering': ['Software Engineer', 'QA Engineer'],
    'Customer Support': ['Support Agent', 'Team Lead'],
    'Sales': ['Account Executive', 'Sales Specialist'],
    'Logistics': ['Coordinator', 'Dispatcher']
}

# Create static employee profiles
employee_pool = []
for emp_id in range(1001, 1001 + num_employees):
    dept = np.random.choice(departments)
    role = np.random.choice(job_roles[dept])
    age = int(np.random.triangular(22, 35, 60))
    # Experience scales roughly with age
    min_exp = max(0, age - 22)
    exp = int(np.random.uniform(0, min(min_exp + 2, 25)))
    
    employee_pool.append({
        'Employee_ID': f"EMP_{emp_id}",
        'Department': dept,
        'Job_Role': role,
        'Age': age,
        'Experience_Years': exp
    })

emp_df_pool = pd.DataFrame(employee_pool)

# Generate 50,000 random employee-date combinations (Workdays)
total_records = 50000
chosen_employees = np.random.choice(emp_df_pool['Employee_ID'], size=total_records)
chosen_dates = np.random.choice(date_range, size=total_records)

df = pd.DataFrame({
    'Employee_ID': chosen_employees,
    'Shift_Date': chosen_dates
})
# Drop duplicates to ensure unique employee-workday records
df = df.drop_duplicates(subset=['Employee_ID', 'Shift_Date']).reset_index(drop=True)

# Top up if duplicates brought us below 50k
while len(df) < total_records:
    needed = total_records - len(df)
    extra_emp = np.random.choice(emp_df_pool['Employee_ID'], size=needed * 2)
    extra_dates = np.random.choice(date_range, size=needed * 2)
    extra_df = pd.DataFrame({'Employee_ID': extra_emp, 'Shift_Date': extra_dates})
    df = pd.concat([df, extra_df]).drop_duplicates(subset=['Employee_ID', 'Shift_Date']).reset_index(drop=True)

df = df.iloc[:total_records]
df = df.merge(emp_df_pool, on='Employee_ID', how='left')
df = df.sort_values(by=['Employee_ID', 'Shift_Date']).reset_index(drop=True)

# ==========================================
# 2. GENERATE FEATURES WITH BUSINESS LOGIC
# ==========================================

# Shift & Temporal features
df['Weekend_Shift'] = df['Shift_Date'].dt.dayofweek.isin([5, 6]).astype(int)
df['Shift_Type'] = np.random.choice(['Morning', 'Evening', 'Night'], size=total_records, p=[0.5, 0.3, 0.2])
df['Shift_Duration_Hours'] = df['Shift_Type'].map({'Morning': 8.0, 'Evening': 8.0, 'Night': 8.5})

# Simulating historical and consecutive sequences via rolling/random proxies
df['Consecutive_Work_Days'] = np.random.choice([1,2,3,4,5,6,7,8], size=total_records, p=[0.15, 0.2, 0.25, 0.2, 0.1, 0.05, 0.03, 0.02])
df['Night_Shifts_Last_30_Days'] = np.where(df['Shift_Type'] == 'Night', np.random.randint(4, 15, size=total_records), np.random.randint(0, 5, size=total_records))

# Overtime Data & Outliers (Extreme Overtime)
base_ot = np.random.choice([0, 1, 2, 3, 4, 5, 6], size=total_records, p=[0.5, 0.2, 0.15, 0.08, 0.04, 0.02, 0.01])
# Inject Outliers (~1% extreme overtime up to 12 hours)
ot_outliers = np.random.uniform(7, 12, size=total_records)
df['Overtime_Hours'] = np.where(np.random.rand(total_records) < 0.01, ot_outliers, base_ot).round(1)
df['Monthly_Overtime_Hours'] = (df['Overtime_Hours'] * np.random.uniform(4, 5.5, size=total_records)).round(1)

# Attendance & Rest (Sleep influenced by Shift and Overtime)
sleep_mu = np.where(df['Shift_Type'] == 'Night', 5.5, 7.0) - (df['Overtime_Hours'] * 0.2)
df['Sleep_Hours_Previous_Night'] = np.random.normal(sleep_mu, 0.8)
# Outliers (Very low sleep)
df['Sleep_Hours_Previous_Night'] = np.where(np.random.rand(total_records) < 0.02, np.random.uniform(2.5, 4.0, size=total_records), df['Sleep_Hours_Previous_Night'])
df['Sleep_Hours_Previous_Night'] = df['Sleep_Hours_Previous_Night'].clip(2.0, 10.0).round(1)

df['Days_Absent_Last_90_Days'] = np.random.negative_binomial(2, 0.4, total_records).clip(0, 15)
df['Late_Arrivals_Last_30_Days'] = np.where(df['Shift_Type'] == 'Night', np.random.randint(0, 5, total_records), np.random.randint(0, 3, total_records))

# Workload Metrics
df['Workload_Intensity'] = np.random.choice(['Low', 'Medium', 'High'], size=total_records, p=[0.25, 0.55, 0.20])
task_map = {'Low': (3, 6), 'Medium': (6, 11), 'High': (11, 18)}
df['Tasks_Assigned'] = df['Workload_Intensity'].map(lambda x: np.random.randint(task_map[x][0], task_map[x][1]))

# Experience buffers stress/workload processing
exp_modifier = (df['Experience_Years'] * 0.02).clip(0, 0.4)
completion_rate = np.where(df['Workload_Intensity'] == 'High', 0.80 + exp_modifier, 0.92 + exp_modifier)
df['Tasks_Completed'] = (df['Tasks_Assigned'] * completion_rate).round().astype(int)
df['Tasks_Completed'] = np.minimum(df['Tasks_Completed'], df['Tasks_Assigned'])

df['Average_Task_Complexity'] = np.random.uniform(1, 5, size=total_records).round(1)
df['Break_Time_Minutes'] = np.where(df['Overtime_Hours'] > 2, np.random.uniform(45, 75), np.random.uniform(30, 45)).round()

# Environmental Conditions
# Adding seasonal variance to temperature based on month
months = df['Shift_Date'].dt.month
base_temp = np.where(months.isin([6, 7, 8]), 28, 18)  # Simple seasonal pattern
df['Temperature_Celsius'] = np.random.normal(base_temp, 5)
# Inject Outliers (Extreme Heat)
df['Temperature_Celsius'] = np.where(np.random.rand(total_records) < 0.01, np.random.uniform(36, 42, size=total_records), df['Temperature_Celsius']).round(1)

df['Humidity_Percentage'] = np.random.normal(55, 12, size=total_records).clip(20, 95).round()
df['Noise_Level_dB'] = np.random.normal(65, 10, size=total_records).clip(40, 95).round()
df['Workspace_Crowding_Score'] = np.random.randint(1, 11, size=total_records)

# Health & Well-being (Stress Model)
# Multi-factor linear proxy for Stress
stress_score = (
    (df['Overtime_Hours'] * 0.4) + 
    (df['Consecutive_Work_Days'] * 0.3) + 
    (df['Shift_Type'].map({'Morning': 0, 'Evening': 1, 'Night': 2.5})) +
    (df['Workload_Intensity'].map({'Low': 1, 'Medium': 3, 'High': 5})) +
    (df['Temperature_Celsius'].map(lambda x: 2 if x > 35 else 0)) +
    (df['Noise_Level_dB'].map(lambda x: 1.5 if x > 80 else 0)) -
    (df['Experience_Years'] * 0.1) +
    np.random.normal(0, 1.5, size=total_records)
)
# Normalize to 1-10 scale
df['Stress_Level'] = np.digitize(stress_score, bins=np.linspace(stress_score.min(), stress_score.max(), 10))

df['Heart_Rate_Avg'] = (65 + (df['Stress_Level'] * 2.5) + np.random.normal(0, 4, size=total_records)).round()
df['Hydration_Breaks_Count'] = np.random.poisson(4, total_records).clip(1, 8)

# Productivity Indicators
efficiency_base = 90 - (df['Stress_Level'] * 2.5) - (df['Overtime_Hours'] * 1.2) + (df['Experience_Years'] * 0.4)
df['Work_Efficiency_Percentage'] = efficiency_base + np.random.normal(0, 4, size=total_records)
df['Work_Efficiency_Percentage'] = df['Work_Efficiency_Percentage'].clip(40, 100).round(1)

error_lambda = np.exp((df['Stress_Level'] * 0.15) + (df['Overtime_Hours'] * 0.1) - (df['Experience_Years'] * 0.03))
df['Error_Count'] = np.random.poisson(error_lambda).clip(0, 12)

df['Response_Time_Minutes'] = (10 + (df['Stress_Level'] * 2.5) + (df['Overtime_Hours'] * 1.5) + np.random.normal(0, 3, size=total_records)).clip(5, 60).round(1)
df['Quality_Score'] = (100 - (df['Error_Count'] * 4) - (df['Stress_Level'] * 2) + np.random.normal(0, 3, size=total_records)).clip(30, 100).round(1)

# ==========================================
# 3. TARGET VARIABLE LOGIC (FATIGUE LEVEL)
# ==========================================
# Weighted Scoring Engine matching rule specifications
fatigue_score = (
    df['Shift_Type'].map({'Morning': 0, 'Evening': 1.5, 'Night': 4.0}) +
    df['Overtime_Hours'].apply(lambda x: 4.5 if x > 4 else (2.5 if x > 2 else 0)) +
    (df['Monthly_Overtime_Hours'] * 0.05) +
    df['Consecutive_Work_Days'].apply(lambda x: 4.0 if x >= 7 else (2.0 if x >= 5 else 0)) +
    df['Sleep_Hours_Previous_Night'].apply(lambda x: 5.0 if x < 5 else (2.5 if x <= 6 else -1.5)) +
    (df['Stress_Level'] * 0.8) +
    df['Workload_Intensity'].map({'Low': -1, 'Medium': 1, 'High': 3.5}) +
    (df['Error_Count'] * 0.6) -
    (df['Work_Efficiency_Percentage'] * 0.05) +
    df['Temperature_Celsius'].apply(lambda x: 1.5 if x > 35 else 0) +
    df['Noise_Level_dB'].apply(lambda x: 1.0 if x > 80 else 0) +
    np.random.normal(0, 2.0, size=total_records)
)

# Enforce crisp class distribution thresholds (Low: 45%, Medium: 35%, High: 20%)
q_low, q_med = np.percentile(fatigue_score, [45, 80])

df['Fatigue_Level'] = np.where(fatigue_score <= q_low, 'Low', 
                               np.where(fatigue_score <= q_med, 'Medium', 'High'))

# Convert to Categorical to lock in the logical distribution
df['Fatigue_Level'] = pd.Categorical(df['Fatigue_Level'], categories=['Low', 'Medium', 'High'], ordered=True)

# ==========================================
# 4. INJECTING MISSING VALUES (2-5%)
# ==========================================
missing_target_cols = [
    'Sleep_Hours_Previous_Night', 
    'Break_Time_Minutes', 
    'Work_Efficiency_Percentage', 
    'Noise_Level_dB', 
    'Average_Task_Complexity'
]

for col in missing_target_cols:
    missing_pct = np.random.uniform(0.02, 0.05)
    mask = np.random.rand(total_records) < missing_pct
    df.loc[mask, col] = np.nan

# ==========================================
# 5. DATA TYPE & FORMAT CONSTRAINTS
# ==========================================
df['Shift_Date'] = df['Shift_Date'].dt.strftime('%Y-%m-%d')

# Save DataFrame to CSV
output_filename = "employee_fatigue_productivity_dataset.csv"
df.to_csv(output_filename, index=False)

print(f"✅ Success! Dataset saved seamlessly as '{output_filename}' with {df.shape[0]} rows and {df.shape[1]} columns.")
print("\nTarget Class Distribution:")
print(df['Fatigue_Level'].value_counts(normalize=True).round(3) * 100)