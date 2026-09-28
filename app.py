import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

#Sera utilizada a package iris do sklearn

iris = load_iris()
X = iris.data
y= iris.target
# os parametros sao atribuidos (X-atributos; y-especie codificada)


#Teste
print("Forma de X:", X.shape)
print("Forma de y:", y.shape)

print("Atributos:", iris.feature_names)
print("Classes:", iris.target_names)
