import pandas as pd
from sklearn.svm import SVC
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


data = pd.read_csv("/Users/siddhartharoynandi/Desktop/train.csv")
'''
data = (data.loc[data['price_range'].isin(['0','3'])])
corr = data.corr()
print corr
fig, ax = plt.subplots(figsize=(21, 21))
colormap = sns.diverging_palette(220, 10, as_cmap=True)
plt.pcolor(corr, cmap='hot')
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()
exit(0)
'''



################## Classification Section #####################################
X = data.drop('price_range',axis=1)
#X = data[['battery_power','px_height','px_width','ram']]
#X =  data[['battery_power','ram']]
Y = data['price_range']
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
X_train, X_test, y_train, y_test = train_test_split(principalDf, Y, test_size=0.33, random_state=101)
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
#print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print accuracy_score(y_test, y_pred)