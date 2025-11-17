import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.regularizers import L2

# Load dataset
dataset = pd.read_csv(r"Churn_Modelling.csv")
print(dataset.head(3))

# Drop useless columns
dataset = dataset.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
print(dataset.head(3))

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset["Gender"] = le.fit_transform(dataset["Gender"])   # Male=1, Female=0

# One-hot encode Geography
dataset = pd.get_dummies(dataset, columns=["Geography"], drop_first=True)

# Split input and output
input_data = dataset.drop("Exited", axis=1)
output_data = dataset["Exited"]

# Scale features
ss = StandardScaler()
input_data = pd.DataFrame(ss.fit_transform(input_data), columns=input_data.columns)

# Build ANN model
ann = Sequential()
ann.add(Dense(10, input_dim=input_data.shape[1], activation="relu",kernel_regularizer=L2(l2=0.01)))
ann.add(Dense(8, activation="relu"))
ann.add(Dense(4, activation="relu"))
ann.add(Dense(1, activation="sigmoid"))

# Compile model
ann.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])


# Train/test split
x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, random_state=42, test_size=0.2)

# Fit model
ann.fit(x_train, y_train, batch_size=100, epochs=10)

prd = ann.predict(x_test)
prd_data = []
for i in prd:
    if i[0] > 0.5 :
        prd_data.append(1)
    else:
       prd_data.append(0)

prd1 = ann.predict(x_train)
prd_data1 = []
for i in prd1:
    if i[0] > 0.5 :
        prd_data1.append(1)
    else:
       prd_data1.append(0)


score = accuracy_score(y_test,prd_data)*100
print(score)
score1 = accuracy_score(y_train,prd_data1)*100
print(score1)

# print(accuracy_score(y_train,prd_data)*100)
