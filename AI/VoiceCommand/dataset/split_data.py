import pandas as pd
from sklearn.model_selection import train_test_split

# Filepath to the CSV file
csv_filepath = r"c:\Users\dangv\Desktop\Multidisciplinary-Project-HK242\AI\VoiceCommand\data\VOICECOMMAND\VOICECOMMAND.csv"

# Load the CSV file into a DataFrame
data = pd.read_csv(csv_filepath)

# Calculate the split ratio
train_ratio = 130 / 155
val_ratio = 25 / 155

# Split the data into training and validation sets
train_data, val_data = train_test_split(data, test_size=val_ratio, random_state=42)

# Save the training and validation sets to separate CSV files
train_data.to_csv(r"c:\Users\dangv\Desktop\Multidisciplinary-Project-HK242\AI\VoiceCommand\data\VOICECOMMAND\train.csv", index=False)
val_data.to_csv(r"c:\Users\dangv\Desktop\Multidisciplinary-Project-HK242\AI\VoiceCommand\data\VOICECOMMAND\val.csv", index=False)

print("Data split completed!")
print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")