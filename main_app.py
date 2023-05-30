import tensorflow.lite as tflite
## import tflite_runtime.interpreter as tflite
from tensorflow import keras
from keras_image_helper import create_preprocessor


import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import os


path = os.path.dirname(__file__)

## get path of css
css_path = path + '/css/custom.css'

## get path of model
model_path = path + '/car-model.tflite'

with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Car Image Prediction service')
st.markdown(
    " Welcome to this simple web application that classifies Cars. There are six classes of Cars - Audi, Hyundai Creta, Mahindra Scorpio, Rolls Royce, Swift , Tata Safari, Toyota Innova to choose from.")
st.markdown("<div style='text-align: center; color: rgb(162, 164, 75);'>Choose a car image from the above classes and have some fun! 🚗</div>",
            unsafe_allow_html=True)



fig = plt.figure()


def predict(file_uploaded):
    with Image.open(file_uploaded) as img:
        img = img.resize((299, 299), Image.Resampling.NEAREST)

    # Load the model
    interpreter = tflite.Interpreter(model_path=model_path)
    # takes the weight of model to the interpreter
    interpreter.allocate_tensors()

    # input into keras
    input_index = interpreter.get_input_details()[0]['index']

    # output for Keras
    output_index = interpreter.get_output_details()[0]['index']

    # from keras
    def preprocess_input(x):
        x /= 127.5
        x -= 1.
        return x

    x = np.array(img, dtype='float32')
    X = np.array([x])

    X = preprocess_input(X)

    ## input is initialized
    interpreter.set_tensor(input_index, X)

    # invoke all the configurations in the neural network
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)

    classes = [
        'Audi',
        'Hyundai Creta',
        'Mahindra Scorpio',
        'Rolls Royce',
        'Swift',
        'Tata Safari',
        'Toyota Innova'
    ]

    results = dict(zip(classes, preds[0]))
    df = pd.DataFrame.from_dict(results, orient='index' , columns=["prediction"])
    df['prediction'] = df['prediction'].astype(float)
    df.sort_values(by="prediction" , ascending=False , inplace=True)
    df['prediction_percentage'] = df['prediction'] .apply(lambda results : "{:.2f} %".format(results))

    max_result_key = max(results, key=results.get)

    result_message = f"Your uploaded Car Image is {max_result_key}"
    return df, result_message


# predict(path=path)

def main():
    file_uploaded = st.file_uploader(
        "Choose File", type=["png", "jpg", "jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)

    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                df, result_message = predict(file_uploaded)
                time.sleep(1)
                st.success('Classified')
                st.dataframe(df.style.highlight_max(color='green', axis=0, subset=["prediction"]))
                st.write(result_message)
                st.pyplot(fig)


if __name__ == "__main__":
    main()
