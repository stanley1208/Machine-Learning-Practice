
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal

# Path of the uploaded file in this environment
file_path='C:/Users/user/Desktop/WenEEG.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

print(df.head())  # Print the first few rows to verify the data


sns.set(font_scale=1.2)

df['Unnamed: 1'] = pd.to_numeric(df['Unnamed: 1'], errors='coerce')
# Define sampling frequency and time vector
sf = 100.  # Sampling frequency in Hz
time = np.arange(len(df)) / sf  # Create a time vector based on the number of samples

# Plot the "EXG Channel 0" signal
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plt.plot(time, df['Unnamed: 1'], lw=1.5, color='k')
plt.xlabel("Time (seconds)")
plt.ylabel("Voltage (EXG Channel 0)")
plt.xlim([time.min(), time.max()])
plt.title('N3 sleep EEG data - EXG Channel 0')
sns.despine()
plt.show()





