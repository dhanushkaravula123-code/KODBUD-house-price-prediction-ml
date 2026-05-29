# Step 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Load dataset
df = pd.read_csv('data.csv')

# Step 3: Show data
print("First 5 rows of dataset:")
print(df.head())

# Step 4: Handle missing values
df = df.dropna()

# Step 5: Convert location (text → numbers automatically)
df = pd.get_dummies(df, columns=['location'])

# Step 6: Separate features and target
X = df.drop('price', axis=1)
y = df['price']

# Step 7: Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 8: Train model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Step 9: Predict on test data
predictions = model.predict(X_test)

print("\nPredictions:", predictions)
print("Actual:", y_test.values)

# Step 10: Visualization
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()

# -------------------------------
# Step 11: User input
print("\n--- Predict Your Own House Price ---")

try:
    area = float(input("Enter area of house: "))
    bedrooms = int(input("Enter number of bedrooms: "))
    loc = input("Enter location (exactly as in dataset): ").lower()

    # Step 12: Create empty input with all columns
    input_data = pd.DataFrame(columns=X.columns)
    input_data.loc[0] = 0

    # Step 13: Fill basic values
    input_data['area'] = area
    input_data['bedrooms'] = bedrooms

    # Step 14: Handle location column
    col_name = 'location_' + loc

    if col_name in input_data.columns:
        input_data[col_name] = 1
    else:
        print("Location not found in dataset!")
        exit()

    # Step 15: Predict
    predicted_price = model.predict(input_data)

    # Step 16: Show results (FIXED INSIDE try block)
    total_price = predicted_price[0]
    price_per_sqft = total_price / area

    print("\nPredicted House Price:", total_price)
    print("Price per sq ft:", price_per_sqft)

except ValueError:
    print("Please enter valid numeric values!")