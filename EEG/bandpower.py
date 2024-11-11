import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.integrate import simps

# Path of the uploaded file in this environment
file_path='C:/Users/user/Desktop/WenEEG.xlsx'
df = pd.read_excel(file_path, engine='openpyxl',skiprows=4)

print(df.head())  # Print the first few rows to verify the data


sns.set(font_scale=1.2)

df['EXG Channel 0'] = pd.to_numeric(df['EXG Channel 0'], errors='coerce')
# Define sampling frequency and time vector
sf = 100.  # Sampling frequency in Hz
time = np.arange(len(df)) / sf  # Create a time vector based on the number of samples

# Plot the "EXG Channel 0" signal
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plt.plot(time, df['EXG Channel 0'], lw=1.5, color='k')
plt.xlabel("Time (seconds)")
plt.ylabel("Voltage (EXG Channel 0)")
plt.xlim([time.min(), time.max()])
plt.title('N3 sleep EEG data - EXG Channel 0')
sns.despine()
plt.show()


# Define window length (4 seconds)
win=4*sf
freqs,psd=signal.welch(df['EXG Channel 0'],sf,nperseg=win)

# Plot the power spectrum
sns.set(font_scale=1.2,style='white')
plt.figure(figsize=(8,4))
plt.plot(freqs,psd,lw=2,color='k')
plt.xlabel("Frequency (Hz)")
plt.ylabel('Power spectral density (V^2/HZ)')
plt.ylim([0,psd.max()*1.1])
plt.title("Welch's periodogram")
plt.xlim([0,freqs.max()])
sns.despine()
plt.show()


# Define delta lower and upper limits
low,high=0.5,4

# Find intersecting values in frequency vector
idx_delta=np.logical_and(freqs>=low,freqs<=high)

# Plot the power spectral density and fill the delta area
plt.figure(figsize=(7,4))
plt.plot(freqs,psd, lw=2,color='k')
plt.fill_between(freqs,psd,where=idx_delta,color='skyblue')
plt.xlabel("Frequency (Hz)")
plt.ylabel('Power spectral density (V^2/HZ)')
plt.xlim([0,10])
plt.ylim([0,psd.max()*1.1])
plt.title("Welch's periodogram")
sns.despine()
plt.show()


# Frequency resolution
freq_res=freqs[1]-freqs[0]  # = 1 / 4 = 0.25

# Compute the absolute power by approximating the area under the curve
delta_power=simps(psd[idx_delta],dx=freq_res)
print("Absolute delta power: %.3f  uV^2"%delta_power)




