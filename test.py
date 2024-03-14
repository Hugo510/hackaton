import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np

try:
    # Cargar el dataset
    df = pd.read_csv('Darknet.CSV', quotechar='"', on_bad_lines='skip')
    #print(df.describe())

    # Convertir las categorías textuales de la columna 'Label' a numéricas
    df['Label'] = pd.Categorical(df['Label'])
    df['Label'] = df['Label'].cat.codes

    # Detectar y eliminar columnas no numéricas
    for col in df.columns:
        if df[col].dtype == 'object':
            df = df.drop(col, axis=1)

    # Eliminar o imputar valores infinitos
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Reemplaza infinitos por NaN
    df.dropna(inplace=True)  # Elimina filas con NaN

    # Opcional: Transformación logarítmica para manejar valores grandes
    # Ten cuidado con esta transformación, ya que no se puede aplicar a valores <= 0
    for col in df.columns:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            if (df[col] > 0).all():  # Solo aplica si todos los valores en la columna son positivos
                df[col] = np.log1p(df[col])  # log1p es log(x + 1) para manejar valores de 0 adecuadamente



    # Continuar con la división en conjuntos de entrenamiento y prueba, y la normalización
    X = df.drop(['Label'], axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


except Exception as e:
    print(f"Error al preparar los datos: {e}")
    exit()

try:
    # Construcción del modelo de red neuronal
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),  # Define la forma de entrada aquí
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


    # Compilar el modelo especificando el optimizador, la función de pérdida y las métricas de evaluación
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo con los datos de entrenamiento
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    # Evaluar el modelo con el conjunto de pruebas para determinar su precisión
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('\nTest accuracy:', test_acc)
except Exception as e:
    print(f"Error durante el entrenamiento o evaluación del modelo: {e}")
    exit()

try:
    # Asegurarse de que la nueva muestra tenga el formato adecuado (por ejemplo, utilizando X_test[0] como muestra)
    nueva_muestra = X_test[0:1]  # Utilizamos slicing para mantener la muestra como un array bidimensional

    # Escalar la nueva muestra utilizando el mismo scaler que para X_train y X_test
    nueva_muestra_escalada = scaler.transform(nueva_muestra)

    # Realizar la predicción con el modelo
    prediccion = model.predict(nueva_muestra_escalada)

    # Imprimir la predicción
    # Dado que el resultado es una probabilidad (debido a la activación sigmoidal), puedes interpretarla según sea necesario.
    # Por ejemplo, puedes considerar un umbral de 0.5 para clasificar entre 0 y 1.
    clase_predicha = (prediccion > 0.5).astype("int32")
    print("Predicción de anomalía (probabilidad):", prediccion)
    print("Clase predicha:", clase_predicha[0][0])  # Imprime 0 o 1 basado en el umbral de 0.5
except Exception as e:
    print(f"Error al realizar la predicción: {e}")
