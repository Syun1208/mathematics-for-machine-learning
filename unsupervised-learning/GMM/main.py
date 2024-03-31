import numpy as np
import pandas as pd
import seaborn as sns
from utils import GMM
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Data loader and preprocesisng
data = load_iris()
target = pd.DataFrame({'Label': data.target.tolist()}, index=range(len(data.target.tolist())))
target.replace({0: 'Iris-Setosa', 1: 'Iris-Versicolour', 2: 'Iris-Virginica'}, inplace=True)
features = pd.DataFrame({'sepal_length': data.data[:,0].tolist(), 'sepal_width': data.data[:,1].tolist(),
                         'petal_length': data.data[:,2].tolist(), 'petal_width': data.data[:,-1].tolist()}, index=range(len(data.data[:, 0].tolist())))

# Data Split
df = pd.concat([features, target], axis=1)
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y= df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modeling
# Choose the best K is 3
gmm = GMM(K=3, n_iters=100)
# gmm.initilize(X_train.to_numpy())
gmm.fit(X_train.to_numpy())

# Prediction
predict = gmm.predict(X_test.to_numpy())

# Evaluation
y_test.replace({'Iris-Setosa': 0, 'Iris-Versicolour': 1, 'Iris-Virginica': 2}, inplace=True)
df_truth_visualize = pd.concat([X_test, y_test], axis=1)
df_predict_visualize = pd.concat([X_test.reset_index(), pd.Series(predict, name='prediction')], axis=1)
df_predict_visualize.drop(columns='index', inplace=True)

silhouette_score(X_test.to_numpy(), predict)
accuracy_score(df_predict_visualize['prediction'], df_truth_visualize['Label'])

# Visualization
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

sns.scatterplot(data=df_predict_visualize, x=df_predict_visualize.columns.tolist()[0], y=df_predict_visualize.columns.tolist()[1], hue='prediction', s=100, ax=ax[0])
ax[0].set_title('The prediction of GMM')
sns.scatterplot(data=df_truth_visualize, x=df_truth_visualize.columns.tolist()[0], y=df_truth_visualize.columns.tolist()[1], hue='Label', s=100, ax=ax[1])
ax[1].set_title('The ground truth of dataset')
plt.savefig('./result.png')
