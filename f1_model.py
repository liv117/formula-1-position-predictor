import pandas as pd
import sklearn

data = pd.read_csv("f1_results.csv")

print(data.head())
print(data.shape)
print(data.dtypes)

X = data[["round", "driver", "constructor", "grid"]]
y = data["position"]

print("\nFeatures:")
print(X.head())
print("\nTarget:")
print(y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), ["driver", "constructor"])
    ],
    remainder="passthrough"
) ##converting the text into numbers so that the model can understand it (training data)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

print("\nTransformed training data shape:", X_train.shape)
print("Transformed testing data shape:", X_test.shape) ##converting the text into numbers so that the model can understand it (testing data)


from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)


from sklearn.metrics import mean_absolute_error 
predictions = model.predict(X_test) 
mae = mean_absolute_error(y_test, predictions) 
print("\nPredictions:") 
print(predictions[:5]) 
print("\nActual positions:") 
print(y_test.head(5).values) # Percentage of predictions within 2 positions 
accuracy = (
     (abs(predictions - y_test.to_numpy()) <= 2).mean() * 100 ) 
print(f"\nAccuracy: {accuracy:.2f}%")
print(f"Mean Absolute Error: {mae:.2f} positions")