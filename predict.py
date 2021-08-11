import cv2
import numpy as np
from matplotlib import pyplot as plt
fname='t2_negative.png'
img = cv2.imread('/content/drive/MyDrive/'+fname)
img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_AREA)
img = img/255.
plt.imshow(img)
plt.show()
img = (np.expand_dims(img,0))
predictions_single = model.predict(img)
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
