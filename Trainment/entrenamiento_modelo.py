import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import pickle
import joblib

dataframe = pd.read_csv('./data/fake_job_postings.csv')  

dataframe = dataframe.loc[:, ['salary_range', 'has_company_logo', 'has_questions', 'employment_type','required_experience', 'required_education', 'fraudulent']]
dataframe = dataframe.dropna()
columnas_problematicas = ['employment_type', 'required_experience', 'required_education']

LabelEncoder = preprocessing.LabelEncoder()
for column in columnas_problematicas:
    dataframe[column] = LabelEncoder.fit_transform(dataframe[column].astype(str))

# 
rangos_de_salario = dataframe['salary_range'].str.split('-', expand=True)
rangos_de_salario = pd.DataFrame({'min_salary': rangos_de_salario[0], 'max_salary': rangos_de_salario[1]})
dataframe = pd.concat([rangos_de_salario, dataframe], axis=1)
dataframe = dataframe.drop(columns=['salary_range'])
#volver a eliminar los valores nulos
dataframe = dataframe.dropna()

#print(dataframe.head())

# Guardar el csv
#dataframe.to_csv('./data/fake_job_postings_clean.csv', index=False)

# seperar los datos en X y Y
x = dataframe.iloc[:, :-1] # Matriz de caracteristicas [min salary, max salary, has_company_logo, has_questions, employment_type, required_experience, required_education]
y = dataframe.iloc[:, -1] # Matriz de etiquetas(respuesta) [fraudulent]

# Estandarizar los datos
scaler = preprocessing.StandardScaler()
x_scaled = scaler.fit_transform(x)

#Separacion de datos en entrenamiento y test
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

randomForest = RandomForestClassifier(max_depth=100)
randomForest.fit(x_train, y_train)

#Visualizacion de los arboles
'''plt.figure(figsize=(12, 12))
tree.plot_tree(randomForest.estimators_[0], filled=True) #para ver los diferentes arboles, cambiar el indice de estimators_ 
plt.show()'''

y_prediccion = randomForest.predict(x_test)

y_prediccion_proba = randomForest.predict_proba(x_test)

'''print(y_prediccion)
print('Probabilidad de ser un fraude: ')
print(y_prediccion_proba)'''

conf_matriz_rf = confusion_matrix(y_test, y_prediccion)

#Visualizacion de la matriz de confunsion
'''plt.figure(figsize=(12,5))
sns.heatmap(conf_matriz_rf, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Actual')
plt.title('Matriz de Confusión - Random Forest')
plt.show()'''

# Guardar el modelo
joblib.dump(randomForest, './Model/random_forest_model.pkl')
joblib.dump(scaler, './Model/scaler.pkl')
joblib.dump(LabelEncoder, './Model/label_encoder.pkl')

nuevo = pd.DataFrame({'min_salary': [2500], 
                          'max_salary': [7000], 
                          'has_company_logo': [1], 
                          'has_questions': [0], 
                          'employment_type': [2], 
                          'required_experience': [5], 
                          'required_education': [4]})

nuevo_scaled = scaler.transform(nuevo)
prediccion = randomForest.predict(nuevo_scaled)
print(str(prediccion))