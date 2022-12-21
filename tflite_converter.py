import tensorflow.lite as tflite
import tensorflow as tf
from tensorflow import keras

best_model = 'xception_v4_1_39_0.954.h5'

model = keras.models.load_model(best_model)
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('car-model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)

print("tensorflow(h5) model converted to tflite")
