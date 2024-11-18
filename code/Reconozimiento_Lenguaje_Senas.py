import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cv2
from tensorflow.keras.models import load_model

# 1. Cargar los puntos clave desde los archivos .pkl
def load_keypoints_from_folder(folder_path):
    data = []
    labels = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pkl'):
            label = file_name.split('_')[0]  # Etiqueta asumida antes del guion bajo
            file_path = os.path.join(folder_path, file_name)

            with open(file_path, 'rb') as f:
                keypoints = pickle.load(f)

            data.append(keypoints)
            labels.append(label)

    return data, labels

# 2. Procesar los puntos clave
def process_keypoints(data):
    processed_data = []

    for video_data in data:
        video_frames = []

        for frame in video_data:
            # Asegurar que las claves existen en el frame
            if 'left_hand' in frame and 'right_hand' in frame:
                left_hand = frame['left_hand']
                right_hand = frame['right_hand']

                # Convertir listas a arrays de NumPy
                left_hand_x = np.array(left_hand['x'])
                left_hand_y = np.array(left_hand['y'])
                right_hand_x = np.array(right_hand['x'])
                right_hand_y = np.array(right_hand['y'])

                # Verificar que no estén vacías
                if left_hand_x.size > 0 and right_hand_x.size > 0:
                    # Concatenar las coordenadas
                    hand_data = np.concatenate([left_hand_x, left_hand_y, right_hand_x, right_hand_y])
                    video_frames.append(hand_data)

        if video_frames:  # Si hay frames válidos
            processed_data.append(np.array(video_frames))

    return processed_data
    processed_data = []

    for video_data in data:
        video_frames = []

        for frame in video_data:
            # Asegurar que las claves existen en el frame
            if 'left_hand' in frame and 'right_hand' in frame:
                left_hand = frame['left_hand']
                right_hand = frame['right_hand']

                # Verificar dimensiones y agregar datos válidos
                if left_hand['x'].size > 0 and right_hand['x'].size > 0:
                    hand_data = np.concatenate([
                        left_hand['x'], left_hand['y'], 
                        right_hand['x'], right_hand['y']
                    ])
                    video_frames.append(hand_data)

        if video_frames:  # Si hay frames válidos
            processed_data.append(np.array(video_frames))

    return processed_data

# 3. Codificar etiquetas
def encode_labels(labels):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    np.save('classes.npy', encoder.classes_)  # Guardar las clases del encoder
    return encoded_labels, encoder

# 4. Rellenar las secuencias
def pad_sequences_data(data, max_len):
    return pad_sequences(data, maxlen=max_len, dtype='float32', padding='post', truncating='post')

# Cargar y procesar los datos
folder_path = '/home/alejandro/Descargas/Base de datos/Keypoints/Keypoints/pkl/Historias_vinetas_2'  # Cambiar a tu carpeta de datos
raw_data, raw_labels = load_keypoints_from_folder(folder_path)
processed_data = process_keypoints(raw_data)
encoded_labels, encoder = encode_labels(raw_labels)

# Determinar longitud máxima
max_len = 50  # Ajustar según el promedio de frames
padded_data = pad_sequences_data(processed_data, max_len)

# Convertir a arrays
X = np.array(padded_data)
y = np.array(encoded_labels)

# 5. Crear el modelo LSTM
def create_lstm_model(input_shape, num_classes):
    model = models.Sequential([
        layers.LSTM(128, activation='relu', input_shape=input_shape, return_sequences=False),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Configurar modelo
input_shape = (X.shape[1], X.shape[2])  # (time_steps, features)
num_classes = len(np.unique(y))
model = create_lstm_model(input_shape, num_classes)

# Resumen del modelo
model.summary()

# 6. Entrenar el modelo
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# 7. Guardar el modelo
model.save('gesture_recognition_lstm_model.keras')

# Cargar el modelo y las clases
model = load_model('gesture_recognition_lstm_model.keras')
encoder.classes_ = np.load('classes.npy', allow_pickle=True)

# Captura de video en tiempo real
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se puede acceder a la cámara.")
    exit()

# Procesar fotogramas para predicción
def process_frame_for_prediction(frame, max_len):
    # Simular datos; debes implementar extracción de keypoints reales
    simulated_data = np.random.rand(max_len, X.shape[2])
    return simulated_data.reshape(1, max_len, X.shape[2])

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede leer el fotograma.")
        break

    # Procesar el fotograma
    sequence = process_frame_for_prediction(frame, max_len)

    # Hacer predicción
    prediction = model.predict(sequence)
    predicted_label = encoder.inverse_transform([np.argmax(prediction)])[0]

    # Mostrar predicción en video
    cv2.putText(frame, f"Predicción: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
