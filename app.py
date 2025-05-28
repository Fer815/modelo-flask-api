from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Inicializar Flask
app = Flask(__name__)

# Cargar modelo TFLite
interpreter = tf.lite.Interpreter(model_path="modelo_convertido.tflite")
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Tamaño de entrada requerido
input_shape = input_details[0]['shape']  # [1, alto, ancho, canales]
target_size = (input_shape[1], input_shape[2])

@app.route("/predecir", methods=["POST"])
def predecir():
    if 'imagen' not in request.files:
        return jsonify({"error": "Falta el archivo de imagen"}), 400

    archivo = request.files["imagen"]
    imagen = Image.open(archivo).convert("RGB")
    imagen = imagen.resize(target_size)
    
    # Convertir a array y normalizar (ajusta según cómo entrenaste el modelo)
    entrada = np.array(imagen, dtype=np.float32) / 255.0
    entrada = np.expand_dims(entrada, axis=0)

    # Cargar la imagen al tensor del modelo
    interpreter.set_tensor(input_details[0]['index'], entrada)
    interpreter.invoke()

    # Obtener salida
    salida = interpreter.get_tensor(output_details[0]['index'])
    clase = int(np.argmax(salida))
    confianza = float(np.max(salida))

    return jsonify({
        "clase": clase,
        "confianza": confianza
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
