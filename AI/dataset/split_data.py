import pandas as pd
from sklearn.model_selection import train_test_split

# File paths for the CSV files
humidity_file = "yolo-humidity-sensor.csv"
temperature_file = "yolo-temperature-sensor.csv"
light_file = "yolo-light-sensor.csv"
movement_file = "yolo-movement-sensor.csv"
# Read the CSV files and keep only the 'value' column
humidity = pd.read_csv(humidity_file, usecols=['value'])
light = pd.read_csv(light_file, usecols=['value'])
movement = pd.read_csv(movement_file, usecols=['value'])
temperature = pd.read_csv(temperature_file, usecols=['value'])

# Merge the data column-wise
merged_data = pd.concat([humidity, light, movement, temperature], axis=1)
merged_data.columns = ['humidity', 'light', 'movement', 'temperature']

# Split the data into training (90%) and validation (10%)
train_data, val_data = train_test_split(merged_data, test_size=1/6, random_state=42)

# Save the split data into CSV files
train_data.to_csv("training.csv", index=False)
val_data.to_csv("validation.csv", index=False)

print("Data has been split and saved into 'training.csv' and 'validation.csv'.")