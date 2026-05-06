import joblib

model = joblib.load("src/model.pkl")

prediction = model.predict([[10]])
print(f"Prediction for input 10: {prediction[0]}")