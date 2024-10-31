import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Specify the path to your Excel file
file_path = 'C:/Users/user/Desktop/Wen EEG .xlsx'

# Read the Excel file
df = pd.read_excel(file_path)

# Optionally save it as a CSV file
df.to_csv('C:/Users/user/Desktop/Wen EEG .csv', index=False)

# Now you can read it as a CSV
df_csv = pd.read_csv('C:/Users/user/Desktop/Wen EEG .csv')

# Display the contents of the DataFrame
print(df_csv)

# Load data
file_path = 'C:/Users/user/Desktop/Wen EEG .csv'
df = pd.read_csv(file_path)
df.rename(columns={'Unnamed: 1': 'Input', 'Unnamed: 8': 'Output'}, inplace=True)

print(df.columns)

# Clean and select "EXG Channel 0" and label columns
df['Input'] = pd.to_numeric(df['Input'], errors='coerce')
df.dropna(subset=['Input', 'Output'], inplace=True)

# Parameters
window_size = 200

# Extract features and labels
X = [
    [
        df['Input'].iloc[i:i + window_size].mean(),
        df['Input'].iloc[i:i + window_size].std(),
        df['Input'].iloc[i:i + window_size].min(),
        df['Input'].iloc[i:i + window_size].max()
    ]
    for i in range(0, len(df) - window_size, window_size)
]
y = [df['Output'].iloc[i] for i in range(0, len(df) - window_size, window_size)]

# Train and evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
accuracy = accuracy_score(y_test, clf.predict(X_test))

print("Classification Accuracy:", accuracy)



