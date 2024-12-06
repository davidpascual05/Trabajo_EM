import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# Paso 1: Crear un conjunto de datos con variables relevantes (simulación de datos)
file_path = "base_de_datos.xlsx"

df = pd.read_excel(file_path)

# Paso 2: Separar variables independientes (X) y dependiente (y)
X = df[[ "Tasa_dependencia", "Tasa_desempleo", "Tasa_natalidad","Crecimiento_Consumo"]]
y = df["Crecimiento_PIB"]

# Calcular Z-score para detectar outliers
df['log_Tasa_desempleo'] = np.log(df['Tasa_desempleo'] + 1)  #Ajustar la variable Tasa_desempleo
df_zscore = df[["Tasa_natalidad", 'log_Tasa_desempleo' ,"Tasa_dependencia", "Crecimiento_Consumo"]].apply(zscore)

# Identificar outliers utilizando Z-score (valores fuera de +/- 2.5 desviaciones estándar)
outliers_zscore = df_zscore[(df_zscore.abs() > 2.5).any(axis=1)]
print("Outliers detectados por Z-Score:")
print(outliers_zscore)

# Visualización de los outliers mediante gráfico de caja y bigotes
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['Tasa_natalidad', 'log_Tasa_desempleo', 'Tasa_dependencia', 'Crecimiento_Consumo']])
plt.title('Boxplot de las Variables')
plt.show()

# Limpiar los datos eliminando los outliers utilizando Z-score (se puede usar el método que prefieras)
df_cleaned_zscore = df[(df_zscore.abs() <= 2.5).all(axis=1)]
print(f'Datos después de eliminar outliers por Z-score: {df_cleaned_zscore.shape[0]} filas.')

# Construcción del modelo de regresión con los datos limpios (usando df_cleaned_zscore o df_cleaned_iqr)
X = df_cleaned_zscore[['Tasa_natalidad', 'log_Tasa_desempleo', 'Tasa_dependencia', 'Crecimiento_Consumo']]
y = df_cleaned_zscore['Crecimiento_PIB']

# Agregar la constante (intercepto) al modelo
X = sm.add_constant(X)

# Ajustar el modelo de regresión
model = sm.OLS(y, X).fit()

# Ver resumen del modelo
print(model.summary())

# Evaluación del modelo
y_pred = model.predict(X)

# Calcular MSE
mse = mean_squared_error(y, y_pred)
print(f'MSE del modelo: {mse}')

# Calcular R2 
r2 = model.rsquared
print(f'R^2 del modelo: {r2}')
