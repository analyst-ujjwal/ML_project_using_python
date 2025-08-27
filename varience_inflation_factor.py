# VARIENCE INFLATION FACTOR (VIF)
# VIF quantifies how much a feature is correlated with other features in a dataset.
# A high VIF indicates that the feature is highly collinear with other features, which can
# lead to instability in regression coefficients and affect the interpretability of the model.

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = pd.read_csv('BMI.csv')
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
X = data[['Gender', 'Height', 'Weight']]
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values,i)
                    for i in range(len(X.columns))]
print(vif_data)


# PRINCIPAL COMPONENT ANALYSIS (PCA)
# PCA is a dimensionality reduction technique that transforms a dataset into a set of orthogonal components
# that capture the maximum variance in the data. It helps to reduce the number of features while
# retaining most of the information.

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# data = {
#     'Height': [170, 165, 180, 175, 160, 172, 168, 177, 162, 158],
#     'Weight': [65, 59, 75, 68, 55, 70, 62, 74, 58, 54],
#     'Age': [30, 25, 35, 28, 22, 32, 27, 33, 24, 21],
#     'Gender': [1, 0, 1, 1, 0, 1, 0, 1, 0, 0]  # 1 = Male, 0 = Female
# }
# df = pd.DataFrame(data)
# print(df)
# # Separate features and target
# x = df.drop('Gender', axis=1)
# y = df['Gender']

# #Scale features
# scaler = StandardScaler()
# x_scaled = scaler.fit_transform(df)

# #PCA
# pca = PCA(n_components = 2)
# x_pca = pca.fit_transform(x_scaled)

# #train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x_pca,y, test_size = 0.3, random_state = 42)

# #Logistic Regression
# model = LogisticRegression()
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)

# #Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize = (5,4))
# sns.heatmap(cm, annot = True,fmt = 'd', cmap = 'Blues', xticklabels = ['Female', 'Male'], yticklabels = ['Female', 'Male'])
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')


# y_numeric = pd.factorize(y)[0]


# plt.figure(figsize= (12,5))
# plt.subplot(1,2,1)
# plt.scatter(x_scaled[:,0], x_scaled[:,1], c=y_numeric, cmap = 'coolwarm', edgecolor = 'k', s=80)
# plt.xlabel('Original Feature 1')
# plt.ylabel('Original Feature 2')
# plt.title('BEFORE PCA: Using First 2 Standorized Features')
# plt.colorbar(label= 'Target classes')

# plt.subplot(1, 2, 2)
# plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y_numeric, cmap='coolwarm', edgecolor='k', s=80)
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('After PCA: Projected onto 2 Principal Components')
# plt.colorbar(label='Target classes')
# plt.tight_layout()
# plt.show()