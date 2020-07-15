import cv2
import numpy as np
from PIL import Image
image = cv2.imread('test_image.jpg')
predict_model = load_model('Face_Recognition.h5')

face = cv2.resize(image, (299,299))
im = Image.fromarray(face, 'RGB')
img_array = np.array(im)
img_array = np.expand_dims(img_array, axis=0)
pred = predict_model.predict(img_array)

print("### Pred ###", pred)

# I trained with five classes of data. If you have more/less amount of classes, feel free to change.
if pred[0][0] > 0.5:
  print("predicted array : ", pred[0][0])
  print(class1)
elif pred[0][1] > 0.5:
  print("predicted array : ", pred[0][1])
  print(class2)
elif pred[0][2] > 0.5:
  print("predicted array : ", pred[0][2])
  print(class3)
elif pred[0][3] > 0.5:
  print("predicted array : ", pred[0][3])
  print(class4)
elif pred[0][4] > 0.5:
  print("predicted array : ", pred[0][4])
  print(class5)
else:
  pass
