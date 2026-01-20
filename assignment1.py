import pandas as pd
from pygam import LinearGAM, s, f

train = pd.read_csv("assignment_data_train.csv")
test = pd.read_csv("assignment_data_test.csv")

x = train[['year', 'month', 'day', 'hour']]
y = train['trips']

model = LinearGAM(s(0) + f(1) + f(2) + f(3))
modelFit = model.gridsearch(x.values, y)

pred = modelFit.predict(test[['year', 'month', 'day', 'hour']].values)