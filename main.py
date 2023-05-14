from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

def load_model():
    global model
    model = tf.keras.models.load_model("vinit.h5")
    

@app.get("/")
async def ping():
    return "Heyy I am alive"

input_shape = (30, 30)

def read_image(image_encoded):
    pil_image = Image.open(BytesIO(image_encoded))
    return pil_image



def preprocess(image):
    imagergb = image.convert('RGB')
    image1 = imagergb.resize((30, 30))
    expand = np.expand_dims(image1, axis=0)
    input_data = np.array(expand) / 255.0
    return input_data



def predict(image):
    # Preprocess the image
    preprocessed_image = preprocess(image)
    
    # Make prediction using the loaded model
    prediction = model.predict(preprocessed_image)
    
    # Perform any additional processing or post-processing if required
    result = prediction.argmax()

    threshold = 0.85
    pred_prob = prediction[0][result]
    return pred_prob, result






@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    # Read the uploaded image
    image = read_image(await file.read())
    
    # Make prediction
    threshold, result = predict(image)
    
    # Convert numpy.int64 to regular Python int
    threshold = int(threshold)
    result = int(result)
    
    # Encode the values
    encoded_value = jsonable_encoder({"threshold": threshold, "result": result})
    
    return encoded_value


if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host='localhost', port=8000)

