from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

# Cargar el modelo
model = load_model('model.h5')  # Asegúrate de que este sea el nombre correcto

# Montar el directorio estático
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Devuelve el archivo index.html desde la carpeta 'static'
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Leer la imagen del archivo
        image = Image.open(BytesIO(await file.read()))
        image = image.convert('L')  # Convertir a escala de grises
        image = image.resize((28, 28))  # Redimensionar a 28x28
        
        # Normalizar la imagen
        image_array = np.array(image) / 255.0  
        image_array = np.expand_dims(image_array, axis=0)  # Agregar dimensión de batch
        image_array = np.expand_dims(image_array, axis=-1)  # Agregar canal de color

        # Hacer la predicción
        prediction = model.predict(image_array)
        
        # Imprimir las predicciones crudas para depuración (opcional)
        print("Predicciones:", prediction)
        
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Devolver la clase predecida en español
        classes = ['Cero', 'Uno', 'Dos', 'Tres', 'Cuatro', 'Cinco', 'Seis', 'Siete', 'Ocho', 'Nueve']
        return {"prediccion": classes[predicted_class]}
    
    except Exception as e:
        return {"error": str(e)}  # Devolver un mensaje de error

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)