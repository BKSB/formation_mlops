import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

hours_data = pd.read_csv("./data/hour.csv")

# yr = 0 -> données d'entraînement
# yr = 1 -> données de test
train_mask = hours_data["yr"] == 0
test_mask = hours_data["yr"] == 1

target_name = "cnt"
features = [
    col for col in hours_data.columns
    if col not in ["casual", "registered", "instant", "dteday", target_name]
]

data_train = hours_data.loc[train_mask, features]
target_train = hours_data.loc[train_mask, target_name]

data_test = hours_data.loc[test_mask, features]
target_test = hours_data.loc[test_mask, target_name]

model = DummyRegressor(strategy="mean")
_ = model.fit(data_train, target_train)

accuracy = model.score(data_test, target_test)


print(f"The test accuracy  is {accuracy:.3f}")