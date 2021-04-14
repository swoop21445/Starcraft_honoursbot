from sklearn.naive_bayes import MultinomialNB
from joblib import dump, load

import csv

with open('opponent_data.csv') as file:
    reader = csv.reader(file)
    data = list(reader)

for state in data:
    if len(state) < 1:
        data.remove(state)
print("blank lines removed done")

y = []
X = []


for state in data:
    state = list(map(int, state))
    category = state[-1]
    y.append(category)
    state.pop(len(state) - 1)
    X.append(state)
print("training fields populated")

model = MultinomialNB()
model.fit(X,y)
print("model created")

dump(model,'bayes_model.jotlib')