import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

dataset = pd.read_csv(r"Perceptron\placement.csv")
print(dataset.head(3))

# sns.scatterplot(x="cgpa",y="resume_score",data=dataset,hue="placed")
# plt.show()

x = dataset.iloc[:,:-1]
y = dataset["placed"]

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=10,test_size=0.2)

pr = Perceptron()
pr.fit(x_train,y_train)

print("PR : ",pr.score(x_train,y_train)*100 , pr.score(x_test,y_test)*100)

plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=pr)
plt.show()

plt.show("")