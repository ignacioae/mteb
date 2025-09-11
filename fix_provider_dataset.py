import pandas as pd

# Leer el archivo CSV
df = pd.read_csv('datasets/provider/catalog/provider_catalog.csv')

# Eliminar la columna 'Unnamed: 0' si existe
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)
    print("Columna 'Unnamed: 0' eliminada")

# Guardar el archivo corregido
df.to_csv('datasets/provider/catalog/provider_catalog.csv', index=False)
print("Archivo provider_catalog.csv corregido")

# También corregir el archivo de test
df_test = pd.read_csv('datasets/provider/test/provider_test.csv')

if 'Unnamed: 0' in df_test.columns:
    df_test = df_test.drop('Unnamed: 0', axis=1)
    print("Columna 'Unnamed: 0' eliminada del archivo de test")

df_test.to_csv('datasets/provider/test/provider_test.csv', index=False)
print("Archivo provider_test.csv corregido")

print("Corrección completada!")
