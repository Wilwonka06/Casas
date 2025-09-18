import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('casas_sucias.csv')
print(df.head())
df.info()
df.describe(include='all')

# 1. Corregir columna superficie
df['superficie'] = df['superficie'].str.replace('m2', '', regex=False).str.strip()
df['superficie'] = df['superficie'].replace('?', np.nan)

# Rellenar valores estimados (por ejemplo, con la mediana)
df['superficie'] = pd.to_numeric(df['superficie'], errors='coerce')
df['superficie'].fillna(df['superficie'].median(), inplace=True)

# 2. Convertir habitaciones a números
df['habitaciones'] = df['habitaciones'].replace({'tres': 3})
df['habitaciones'] = pd.to_numeric(df['habitaciones'], errors='coerce')
df['habitaciones'].fillna(df['habitaciones'].median(), inplace=True)

# 3. Transformar antiguedad
df['antiguedad'] = df['antiguedad'].replace({'nueva': 0})
df['antiguedad'] = pd.to_numeric(df['antiguedad'], errors='coerce')
df['antiguedad'] = df['antiguedad'].abs() 

# 4. Normalizar ubicacion corrigiendo errores tipográficos y nulos
ubicacion_map = {
    'urbnaa': 'urbano',
    'rurall': 'rural',
    'urbano': 'urbano',
    'rural': 'rural'
}
df['ubicacion'] = df['ubicacion'].str.lower().map(ubicacion_map)

# Convertir ubicacion a 0 y 1  (rural=0, urbano=1)
df['ubicacion'] = df['ubicacion'].replace({'rural': 0, 'urbano': 1})
df['ubicacion'] = pd.to_numeric(df['ubicacion'], errors='coerce')
# 5. Detectar y tratar precios anómalos
df.loc[df['precio'] < 0, 'precio'] = np.nan
df.loc[df['precio'] > 1000000, 'precio'] = np.nan
df['precio'].fillna(df['precio'].median(), inplace=True)

#verificamos si hay valores nulos
print('datos nulos por columna:')
print(df.isnull().sum())
#verificamos si hay Nan

print('datos NaN por columna:')
print(df.isna().sum())

# Verificamos si hay duplicados
print('duplicados:')
print(df.duplicated().sum())

#Borramos duplicados  y datos NaN
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

print(df.head(20))
print(df.info())

# Guardamos el DataFrame limpio
df.to_csv('casas_limpias.csv', index=False)