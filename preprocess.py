import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# 📊 Load the dataset
file_path = "HDFCBANK_HOURLY_INDICATORS_20250326_015226.csv"   # Your data file
df = pd.read_csv(file_path, index_col=0, parse_dates=True)

# ✅ Sort by timestamp
df.sort_index(inplace=True)

# 🛠️ Handle missing values for SMA_20
# Fill missing SMA_20 with the first available valid value (forward fill)
df['SMA_20'].fillna(method='ffill', inplace=True)

# For safety, fill any remaining NaNs with the column mean
df.fillna(df.mean(), inplace=True)

# 🔥 Select relevant columns
cols_to_scale = ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA_20', 'EMA_20', 'Price']

# 🚀 Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# 📊 Split the data into train, validation, and test sets
train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.15)

train = df.iloc[:train_size]
val = df.iloc[train_size:train_size + val_size]
test = df.iloc[train_size + val_size:]

# 🛠️ Save the preprocessed data in your MacroHFT project
output_dir = "../macroHFT_indian_equity/Data/"
train.to_csv(f"{output_dir}HDFCBANK_train.csv")
val.to_csv(f"{output_dir}HDFCBANK_val.csv")
test.to_csv(f"{output_dir}HDFCBANK_test.csv")

# ✅ Confirm the process completion
print("✅ Data preprocessed and saved successfully!")
print(f"📊 Train: {len(train)} rows")
print(f"📈 Validation: {len(val)} rows")
print(f"📉 Test: {len(test)} rows")
