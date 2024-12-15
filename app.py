from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
import matplotlib
matplotlib.use('Agg')  # Usar el backend 'Agg'
from io import BytesIO
from base64 import b64encode
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from flask import Flask, render_template, request, redirect, url_for
from sklearn.metrics import mean_squared_error, r2_score


app = Flask(__name__)

# Página principal con el navbar
@app.route('/')
def index():
    return render_template('index.html')

# Ejercicio Base: Predicción de Demanda y Gráfico
@app.route('/ejercicio_base', methods=['GET', 'POST'])
def ejercicio_base():
    # Generación de datos ficticios de ventas
    data = {
        'Semana': np.arange(1, 11),
        'Ventas_Pasadas': [50, 55, 53, 60, 62, 65, 70, 75, 80, 85],
        'Demanda': [52, 57, 55, 63, 65, 68, 72, 78, 83, 88]
    }
    df = pd.DataFrame(data)

    # División en variables predictoras (X) y etiqueta (y)
    X = df[['Ventas_Pasadas']]
    y = df['Demanda']

    # Entrenamiento del modelo de regresión lineal
    model = LinearRegression()
    model.fit(X, y)

    # Valor por defecto para predicción
    prediccion = None
    nueva_venta = None

    # Si se recibe un valor desde el formulario
    if request.method == 'POST':
        nueva_venta = float(request.form['nueva_venta'])
        prediccion = model.predict([[nueva_venta]])[0]

    # Gráfico de la regresión lineal
    fig, ax = plt.subplots()
    ax.scatter(df['Ventas_Pasadas'], df['Demanda'], color='blue', label='Datos reales')
    ax.plot(df['Ventas_Pasadas'], model.predict(X), color='red', label='Modelo ajustado')
    ax.set_xlabel('Ventas Pasadas')
    ax.set_ylabel('Demanda')
    ax.legend()

    # Convertir la gráfica a base64 para mostrar en el navegador
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    graph_url = base64.b64encode(output.getvalue()).decode('utf8')

    return render_template('ejercicio_base.html', prediccion=prediccion, nueva_venta=nueva_venta, graph_url=graph_url)

# Datos ficticios con nuevas variables
data = {
    'Semana': np.arange(1, 11),
    'Ventas_Pasadas': [50, 55, 53, 60, 62, 65, 70, 75, 80, 85],
    'Estacionalidad': [1, 1, 2, 2, 3, 3, 4, 4, 1, 1],  # 1 = invierno, 2 = primavera, etc.
    'Promocion': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],  # 0 = no promoción, 1 = promoción activa
    'Demanda': [52, 57, 55, 63, 65, 68, 72, 78, 83, 88]
}
df = pd.DataFrame(data)

# Variables predictoras (Ventas Pasadas, Estacionalidad, Promoción)
X = df[['Ventas_Pasadas', 'Estacionalidad', 'Promocion']]
y = df['Demanda']  # Variable dependiente

# Crear el modelo de regresión lineal
model = LinearRegression()
model.fit(X, y)

@app.route('/mejora1', methods=['GET', 'POST'])
def mejora1():
    prediccion = None
    nueva_venta = None
    estacionalidad = None
    promocion = None
    img = None

    if request.method == 'POST':
        # Obtener los valores ingresados por el usuario
        nueva_venta = float(request.form['ventas_pasadas'])
        estacionalidad = int(request.form['estacionalidad'])
        promocion = int(request.form['promocion'])

        # Realizar la predicción usando el modelo entrenado
        prediccion = model.predict([[nueva_venta, estacionalidad, promocion]])[0]
        
        # Generar la gráfica con Matplotlib
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X['Ventas_Pasadas'], y, color='blue', label='Datos reales')
        ax.plot(X['Ventas_Pasadas'], model.predict(X), color='red', label='Modelo ajustado')
        ax.scatter(nueva_venta, prediccion, color='green', zorder=5)
        ax.text(nueva_venta, prediccion, f'Predicción: {prediccion:.2f}', fontsize=12, color='green')
        ax.set_xlabel('Ventas Pasadas')
        ax.set_ylabel('Demanda')
        ax.legend()

        # Guardar la gráfica en formato base64 para mostrarla en la página
        img_io = io.BytesIO()
        fig.savefig(img_io, format='png')
        img_io.seek(0)
        img = base64.b64encode(img_io.getvalue()).decode('utf-8')
        plt.close(fig)

    return render_template('mejora1.html', prediccion=prediccion, nueva_venta=nueva_venta, estacionalidad=estacionalidad, promocion=promocion, img=img)



# Verificar si el directorio de carga existe
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Función para cargar el CSV
def cargar_datos_csv(archivo):
    try:
        df = pd.read_csv(archivo)
        print(f"Datos cargados: {df.head()}")  # Imprimir las primeras filas para ver si cargaron bien
        return df
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return None

@app.route('/mejora2', methods=['GET', 'POST'])
def mejora2():
    predicciones = []
    img = None
    csv_data = None
    error_msg = None

    if request.method == 'POST' and 'csv_file' in request.files:
        archivo_csv = request.files['csv_file']
        
        if archivo_csv.filename.endswith('.csv'):
            try:
                archivo_path = os.path.join(app.config['UPLOAD_FOLDER'], archivo_csv.filename)
                archivo_csv.save(archivo_path)

                # Cargar los datos del CSV
                csv_data = cargar_datos_csv(archivo_path)

                if csv_data is None or csv_data.empty:
                    error_msg = "Hubo un problema al cargar los datos del archivo CSV o el archivo está vacío."
                else:
                    # Preprocesar los datos
                    X = csv_data[['Ventas_Pasadas', 'Estacionalidad', 'Promocion']]
                    y = csv_data['Demanda']

                    # Entrenamos el modelo con los datos del CSV
                    model = LinearRegression()
                    model.fit(X, y)

                    # Realizamos las predicciones
                    predicciones = model.predict(X)

                    # Generamos la gráfica
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(csv_data['Ventas_Pasadas'], y, color='blue', label='Datos reales')
                    ax.plot(csv_data['Ventas_Pasadas'], predicciones, color='red', label='Modelo ajustado')

                    # Mostrar las predicciones sobre la gráfica
                    for i in range(len(csv_data)):
                        ax.text(csv_data['Ventas_Pasadas'][i], predicciones[i], f'{predicciones[i]:.2f}', fontsize=10, color='green')

                    ax.set_xlabel('Ventas Pasadas')
                    ax.set_ylabel('Demanda')
                    ax.legend()

                    # Guardar la gráfica en formato base64
                    img_io = io.BytesIO()
                    fig.savefig(img_io, format='png')
                    img_io.seek(0)
                    img = base64.b64encode(img_io.getvalue()).decode('utf-8')
                    plt.close(fig)

            except Exception as e:
                error_msg = f"Error al procesar el archivo CSV: {e}"
        else:
            error_msg = "Por favor, cargue un archivo CSV válido."

    return render_template('mejora2.html', predicciones=predicciones, img=img, csv_data=csv_data, error_msg=error_msg)


# Datos de prueba para el ejercicio
data = {
    'Semana': np.arange(1, 11),
    'Ventas_Pasadas': [50, 55, 53, 60, 62, 65, 70, 75, 80, 85],
    'Demanda': [52, 57, 55, 63, 65, 68, 72, 78, 83, 88],
    'Promocion': [0, 0, 0, 1, 0, 0, 1, 0, 0, 1]  # Variable adicional: Promoción
}

df = pd.DataFrame(data)

# Función para crear la imagen codificada en base64 para mostrarla en HTML
def create_image(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

# Ruta de la mejora3
@app.route('/mejora3', methods=['GET', 'POST'])
def mejora3():
    error_msg = None
    predicciones = []
    img = None
    img_residual = None
    img_hist = None

    if request.method == 'POST':
        try:
            # División de los datos en características (X) y etiquetas (y)
            X = df[['Ventas_Pasadas', 'Promocion']]  # Variables predictoras
            y = df['Demanda']  # Variable dependiente (Demanda)

            # Crear y entrenar el modelo de regresión lineal
            model = LinearRegression()
            model.fit(X, y)

            # Predicción sobre el conjunto de datos (puedes cambiar a datos de prueba si es necesario)
            y_pred = model.predict(X)

            # Evaluación del modelo
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # Predicción para un nuevo dato de ejemplo
            nueva_venta = [[90, 1]]  # Ejemplo de predicción para ventas pasadas=90 y promoción=1
            prediccion = model.predict(nueva_venta)

            # Crear gráficos
            # Gráfico de predicciones
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(df['Semana'], y, color='blue', label='Datos Reales')
            ax.plot(df['Semana'], y_pred, color='red', label='Modelo Ajustado')
            ax.set_xlabel('Semana')
            ax.set_ylabel('Demanda')
            ax.set_title('Predicción vs Datos Reales')
            ax.legend()
            img = create_image(fig)

            # Gráfico de residuos
            fig_residual, ax_residual = plt.subplots(figsize=(8, 6))
            ax_residual.scatter(df['Semana'], y - y_pred, color='purple')
            ax_residual.axhline(y=0, color='black', linestyle='--')
            ax_residual.set_xlabel('Semana')
            ax_residual.set_ylabel('Residual')
            ax_residual.set_title('Gráfico de Residuos')
            img_residual = create_image(fig_residual)

            # Histograma de errores
            fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
            ax_hist.hist(y - y_pred, bins=10, color='green')
            ax_hist.set_xlabel('Error')
            ax_hist.set_ylabel('Frecuencia')
            ax_hist.set_title('Histograma de Errores')
            img_hist = create_image(fig_hist)

            # Mostrar predicción para nuevo dato
            predicciones = [f"{prediccion[0]:.2f}"]

        except Exception as e:
            error_msg = str(e)

    return render_template('mejora3.html', 
                           error_msg=error_msg,
                           predicciones=predicciones,
                           img=img,
                           img_residual=img_residual,
                           img_hist=img_hist)

if __name__ == '__main__':
    app.run(debug=True)
