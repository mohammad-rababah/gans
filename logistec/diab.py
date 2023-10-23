import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('diabetes.csv')
train, test = train_test_split(data, test_size=20, shuffle=True)
X_train = train.drop(columns=['Outcome'], axis=1)
Y_train = train['Outcome']
model = LogisticRegression(max_iter=100000)
model.fit(X_train, Y_train)
X_test = test.drop(columns=['Outcome'], axis=1)
Y_test = test['Outcome']
print(model.score(X_test, Y_test))
