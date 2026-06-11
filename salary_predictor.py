# Load Dataset
import pandas as pd
df = pd.read_csv("employees.csv")

# Analyze Dataset
print(df.head())
print(df.info())
print(df.describe())

# Visualize Dataset
import matplotlib.pyplot as plt
plt.scatter(df["Experience"], df["Salary"])
plt.title("Salary Vs. Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

# Mark Features & Lablels
x = df[["Experience"]]
y = df["Salary"]

print(x)
print(y)

# Train/Test Split
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.2, random_state=42)

# Train Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain, ytrain)

# Make Predictions

predictions = model.predict(xtest)
print(predictions)

# Evaluate Model
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y, model.predict(x))
print(f"mae: {mae}")

# Prediction Line
plt.scatter(x, y)
plt.plot(x, model.predict(x))
plt.show()

# For user input
experiance = float(input("Enter Experience in years:\n"))
new_data = pd.DataFrame({"Experience": [experiance]})
expected_salary = model.predict(new_data)
print(f"Your predicted Salary is: {expected_salary[0]}")