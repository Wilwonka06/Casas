import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# 1. Cargar datos
df = pd.read_csv("casas_limpias.csv")
print("Datos cargados:")
print(df.head(25))

# 2. convertir variables categóricas a numéricas
df = pd.get_dummies(df, columns=["ubicacion",], drop_first=True)

# 3. Definir características y etiqueta
X = df.drop(columns=["precio"])
Y= df["precio"]

# 4. Dividir datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# 5. Creamos el arbol de decisión
modelo = DecisionTreeRegressor(max_depth=3, random_state=42)
modelo.fit(X_train, y_train)

# 6. Evaluamos 
y_pred = modelo.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

""" #creamoe el diagrama del árbol (opcional)
plt.figure(figsize=(12,12))
plot_tree(modelo, feature_names=X.columns, filled=True)
plt.show()
 """
# 7. Guardar el modelo y las columnas
joblib.dump(modelo, "modelo_precio_casas.pkl")
joblib.dump(X.columns.tolist(), "columnas_modelo_precio_casas.pkl")