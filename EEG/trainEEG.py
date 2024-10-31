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
# Fill NaN values in 'Input' with a placeholder by directly assigning back to the column
df['Input'] = df['Input'].fillna('Unknown')

# Convert 'Output' to numeric, forcing any non-numeric values to NaN
df['Output'] = pd.to_numeric(df['Output'], errors='coerce')

# Drop rows where 'Output' is NaN after conversion
df.dropna(subset=['Output'],inplace=True)

# print(df.columns)




X=pd.get_dummies(df[['Input']])
y=df['Output']

# Train and evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
accuracy = accuracy_score(y_test, clf.predict(X_test))

print("Classification Accuracy:", accuracy)



