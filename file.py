
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#improve readability by ffixing the display format for floating numbers to 2 decimal places
pd.options.display.float_format = '{:.2f}'.format
#read dataset
df = pd.read_excel('dataset.xlsx')
### ## Exploration & preprocessing
#display dimension
df.shape
#-------------------Data Exploration---------------------------
#visualize the first lines
df.head()
#display the last rows
df.tail()
#provides a concise summary of the DataFrame df
df.info()
#generates descriptive statistics for numerical columns
df.describe()
#number of unique values in each column
dict_unique = {}
for col in df.columns:
    dict_unique[col] = len(df[col].unique())
dict_unique
#frequency of unique values in each column
dict_value_counts = {}
for i in df.columns.to_list():
    dict_value_counts[i] = df[i].value_counts()
dict_value_counts
# Duplicate records analysis
df.duplicated().sum()
#visualize missing values
sns.heatmap(df.isnull(), cbar=False)
#number and percentage of missing values
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_data = pd.DataFrame({'missing_values': missing_values, 'missing_percentage': missing_percentage})
print(missing_data)
#percentage of populated non-null values for each column
dict_populated = {}
for i in df.columns.to_list():
    dict_populated[i] = df[i].count()/len(df)*100
dict_populated
#display transactions with amount equal zero
df[df['Amount'] == 0]
#identify numirical values
numeric_columns = df.select_dtypes(include=[np.number]).columns
print(numeric_columns)
# Calculate correlation matrix for numeric columns 
corr_matrix = df[numeric_columns].corr()
# Plot the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix (Numeric Columns)')
plt.show()
#number of frauds
fraud_cases=len(df[df['Fraud']==1])
print(' Number of Fraud Cases:',fraud_cases)
#number of non-fraud
non_fraud_cases=len(df[df['Fraud']==0])
print('Number of Non Fraud Cases:',non_fraud_cases)
fraud=df[df['Fraud']==1]
genuine=df[df['Fraud']==0]
fraud.Amount.describe()
genuine.Amount.describe()
df.hist(figsize=(20,20),color='purple')
plt.show()
# Transformer la colonne 'Date' en une valeur numérique
df['Date'] = pd.to_datetime(df['Date']).astype(int) / 10**9
# Encodage des variables catégorielles
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for column in ['Merchnum', 'Merch description', 'Merch state', 'Transtype']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le
# Remplir les valeurs manquantes avec la médiane des colonnes
df.fillna(df.median(), inplace=True)
### ### Modeling & Evaluation
# Model1: RandomForestClassifier
# separate explicative variables X and tagret Y
X = df.drop(['Fraud'], axis=1)
Y = df['Fraud']
#split data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)
# forming the RandomForestClassifier model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
model = rfc.fit(X_train, Y_train)
# the variable importance
importances = model.feature_importances_
feature_names = X.columns
# trace the variables importance
plt.figure(figsize=(10, 8))
sns.barplot(x=importances, y=feature_names)
plt.title('Importance des Variables')
plt.show()
# Prediction
prediction = model.predict(X_test)
#calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,prediction)
# Évaluation du modèle
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test, prediction))
print(classification_report(Y_test, prediction))
# Model 2 : LogisticRegression
X = df.drop(['Fraud'], axis=1)
Y = df['Fraud']
from sklearn.model_selection import train_test_split
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y, test_size=0.30, random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
model2=lr.fit(X1_train,Y1_train)
prediction2=model2.predict(X1_test)
# Compte The occurrences of each class in Y_test et prediction
test_counts = Y_test.value_counts()
pred_counts = pd.Series(prediction2).value_counts()

#visualize
plt.figure(figsize=(8, 6))
sns.barplot(x=test_counts.index, y=test_counts.values, color='blue', alpha=0.5, label='True')
sns.barplot(x=pred_counts.index, y=pred_counts.values, color='red', alpha=0.5, label='Predicted')

plt.title('Comparaison des vraies valeurs et des prédictions')
plt.xlabel('Classe de fraude')
plt.ylabel('Nombre de transactions')
plt.legend()
plt.show()
#model evaluation
print("Confusion matrix :\n", confusion_matrix(Y1_test, prediction2))
print("Classification report :\n", classification_report(Y1_test, prediction2))
accuracy_score(Y1_test,prediction2)
# Model 3 : DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
X2 = df.drop(['Fraud'], axis=1)
Y2 = df['Fraud']
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.3,)
dt = DecisionTreeRegressor()
model3 = dt.fit(X2_train, Y2_train)
prediction3 = model3.predict(X2_test)
# compare predictions with reel values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=Y2_test, y=prediction3, alpha=0.6)
plt.xlabel('Reel vlaues')
plt.ylabel('Predictions')
plt.title('Compare reel and predeted values')
plt.plot([Y2_test.min(), Y2_test.max()], [Y2_test.min(), Y2_test.max()], color='red', linestyle='--')
plt.show()
#evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(Y2_test, prediction3)
mse = mean_squared_error(Y2_test, prediction3)
r2 = r2_score(Y2_test, prediction3)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

accuracy_score(Y2_test,prediction3)
# This is a sample Python file

