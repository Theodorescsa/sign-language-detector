import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

with open('./data_42.pickle', 'rb') as f:
    data_42_dict = pickle.load(f)

with open('./data_84.pickle', 'rb') as f:
    data_84_dict = pickle.load(f)

data_42 = data_42_dict['data']
labels_42 = data_42_dict['labels']

data_84 = data_84_dict['data']
labels_84 = data_84_dict['labels']


processed_data_42 = [sample + [0] * 42 for sample in data_42]

data_combined = np.asarray(processed_data_42 + data_84)
labels_combined = np.asarray(labels_42 + labels_84)

x_train, x_test, y_train, y_test = train_test_split(data_combined, labels_combined, test_size=0.2, shuffle=True, stratify=labels_combined)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)
print(f"{score * 100}% phân loại thành công")

with open("model.p", 'wb') as f:
    pickle.dump({'model': model}, f)
