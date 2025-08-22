import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from google.colab import files

model = load_model("shakespeare_face_cnn.h5")
uploaded = files.upload()   
test_path = list(uploaded.keys())[0]
img = load_img(test_path, target_size=(128,128))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)  
x = x / 255.0 
  
pred = model.predict(x)[0][0]
if pred > 0.5:
    label = "Shakespeare"
else:
    label = "Not Shakespeare"
plt.imshow(load_img(test_path))
plt.title(f"Prediction: {label} ({pred:.2f})")
plt.axis("off")
plt.show()
