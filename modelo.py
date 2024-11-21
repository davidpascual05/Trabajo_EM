import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# Cargar el DataFrame
data = {
    "Tasa_natalidad": [21.7, 21.2, 21.1, 21.4, 22.0, 21.1, 20.7, 20.7, 20.1, 19.9, 19.5, 19.6, 19.4, 19.2, 19.5, 18.7, 18.7, 18.0, 17.3, 16.2, 15.2, 14.1, 13.6, 12.7, 12.3, 11.9, 11.4, 11.0, 10.8, 10.5, 10.3, 10.2, 10.1, 9.4, 9.1, 9.1, 9.2, 9.1, 9.4, 9.9, 10.1, 10.4],
   "Tasa_dependencia": [17.909, 18.03653, 18.17481, 18.32798, 18.55604, 18.88002, 19.28547, 19.71823, 20.14523, 20.55782, 20.9195, 21.23882, 21.56587, 21.91972, 22.32019, 22.73626, 23.1622, 23.61136, 24.00402, 24.37007, 24.78977, 24.94853, 24.73435, 24.35321, 24.19725, 24.17281, 23.98849, 24.02518, 24.38103, 24.89441, 25.41763, 25.96843, 26.7229, 27.50276, 28.05225, 28.48605, 28.92358, 29.29925, 29.5815, 30.16036, 30.7339, 31.51368],
    "Tasa_desempleo": [ 11.4, 14.17, 16, 17.49, 20.25, 21.64, 21.259, 20.607, 19.854, 17.331, 16.273, 15.929, 17.697, 22.156, 24.209, 22.675, 22.142, 20.698, 18.673, 15.475, 13.785, 10.348, 11.146, 11.283, 11.09, 9.146, 8.452, 8.232, 11.255,  19.86, 21.391, 24.789, 26.094, 24.441, 22.057, 19.635, 17.224, 15.255, 14.105, 14.781, 12.917, 12.179],
    "Crecimiento_PIB": [ 2.208728107, -0.132468452, 1.246461606, 1.77011574, 1.784687507, 2.321435946, 3.253321706, 5.547122619, 5.094324204, 4.827030271, 3.781393555, 2.546000585, 0.9292153161, -1.031491729, 2.383195326, 2.757494046, 2.660561604, 3.702494627, 4.393100735, 4.490553378, 5.245943824, 3.933002765, 2.730948329, 2.981936316, 3.122733517, 3.652084192, 4.102685741, 3.604738462, 0.8870670456, 0.1629195178, -0.8143847766, -2.958922132, -1.403341868, 1.395775367, 3.838518877, 3.037774142, 2.975760729, 2.284469419, 1.983966223, 6.403173615, 5.770647676, 2.50329436],
   "Crecimiento_Consumo" : [
     2.065750639, -0.999108138, 0.040096956,
    0.388835524, -0.199109144, 2.283413411, 3.402753073, 5.952281543, 4.891369805,
    5.426634212, 3.510500381, 2.886757617, 2.173510668, -1.902640722, 1.084463017,
    1.713573089, 2.451108155, 2.828895571, 4.408452627, 4.798158279, 4.436379088,
    3.87576499, 3.076414455, 2.365957269, 4.057202019, 4.13318758, 3.984850747,
    3.394328479, -0.702036974, 0.369345683, -2.461965343, -3.318272489,
    -2.887557852, 1.689365965, 2.947542435, 2.693525226, 3.023996044, 1.694760191,
    1.08176108, 7.064013622, 4.706664409, 1.7777758
]

}
# Importar las bibliotecas necesarias

df = pd.DataFrame(data)


# Calcular Z-score para detectar outliers
df['log_Tasa_desempleo'] = np.log(df['Tasa_desempleo'] + 1)
df_zscore = df[["Tasa_natalidad", 'log_Tasa_desempleo' ,"Tasa_dependencia", "Crecimiento_Consumo"]].apply(zscore)

# Identificar outliers utilizando Z-score (valores fuera de +/- 3 desviaciones estándar)
outliers_zscore = df_zscore[(df_zscore.abs() > 3).any(axis=1)]
print("Outliers detectados por Z-Score:")
print(outliers_zscore)

# Calcular Q1, Q3 y IQR para detectar outliers mediante el método IQR
Q1 = df[['Tasa_natalidad', 'log_Tasa_desempleo', 'Tasa_dependencia', 'Crecimiento_Consumo']].quantile(0.25)
Q3 = df[['Tasa_natalidad', 'log_Tasa_desempleo', 'Tasa_dependencia', 'Crecimiento_Consumo']].quantile(0.75)
IQR = Q3 - Q1

# Detectar outliers con IQR
outliers_iqr = df[((df[['Tasa_natalidad', 'log_Tasa_desempleo', 'Tasa_dependencia', 'Crecimiento_Consumo']] < (Q1 - 1.5 * IQR)) |
                   (df[['Tasa_natalidad', 'log_Tasa_desempleo', 'Tasa_dependencia', 'Crecimiento_Consumo']] > (Q3 + 1.5 * IQR))).any(axis=1)]
print("Outliers detectados por IQR:")
print(outliers_iqr)

# Visualización de los outliers mediante Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['Tasa_natalidad', 'log_Tasa_desempleo', 'Tasa_dependencia', 'Crecimiento_Consumo']])
plt.title('Boxplot de las Variables')
plt.show()

# Limpiar los datos eliminando los outliers utilizando Z-score (se puede usar el método que prefieras)
df_cleaned_zscore = df[(df_zscore.abs() <= 3).all(axis=1)]
print(f'Datos después de eliminar outliers por Z-score: {df_cleaned_zscore.shape[0]} filas.')

# Limpiar los datos eliminando los outliers utilizando IQR
df_cleaned_iqr = df[~((df[['Tasa_natalidad', 'log_Tasa_desempleo', 'Tasa_dependencia', 'Crecimiento_Consumo']] < (Q1 - 1.5 * IQR)) |
                      (df[['Tasa_natalidad', 'log_Tasa_desempleo', 'Tasa_dependencia', 'Crecimiento_Consumo']] > (Q3 + 1.5 * IQR))).any(axis=1)]
print(f'Datos después de eliminar outliers por IQR: {df_cleaned_iqr.shape[0]} filas.')

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

# Calcular MSE (Error Cuadrático Medio)
mse = mean_squared_error(y, y_pred)
print(f'MSE del modelo: {mse}')

# Calcular R2 (Coeficiente de Determinación)
r2 = model.rsquared
print(f'R^2 del modelo: {r2}')

