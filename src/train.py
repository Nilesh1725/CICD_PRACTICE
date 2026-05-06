from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib

# Dummy dataset
data = pd.DataFrame({
    "x": [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
    "y": [2, 4, 6, 8, 10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42]
})

X = data[["x"]]
y = data["y"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "src/model.pkl")
print("Model trained and saved!")