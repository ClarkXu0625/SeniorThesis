import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
data = pd.read_csv('wrong_predictions.csv')

print(data['75'])

def plot_wave():
    # Plotting the first few waveforms to check the data
    plt.figure(figsize=(12, 7))  # Set the figure size for better visibility

    # Loop through the first few rows to plot multiple waveforms
    for index, row in data.head().iterrows():
        plt.plot(row[:-2], label=f'Waveform {index+1} (Label: {row["TrueLabel"]}, Pred: {row["Prediction"]})')  # Assuming last two columns are labels

    plt.title('Waveforms of Incorrect Predictions')
    plt.xlabel('Time or Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
