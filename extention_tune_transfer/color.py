# kc3143
import numpy as np
from PIL import Image
import cv2
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
from keras.preprocessing.image import img_to_array, load_img

color_x = 174
color_y = 281
data_x,data_y  = 361,641

# def csv2image(csvfile,x,y):
#     im = Image.new("RGB",(x,y))
#     with open(csvfile) as f:
#         for i in range(x):
#             for j in range(y):
#                 line = f.readline()
#                 rgb = line.split()[0].split(",")
#                 im.putpixel((i,j),(int(rgb[0]),int(rgb[1]),int(rgb[2])))
#     imsave("input.png", im)


image = img_to_array(load_img('test_train3.jpeg'))
# image = img_to_array(csv2image('color.csv',color_x,color_y))
# image = csv2image('color.csv',color_x,color_y)
image = cv2.resize(image, (360, 360), interpolation=cv2.INTER_CUBIC)
image = np.array(image, dtype=float)
# print ("image =",image)


X = rgb2lab(1.0/255*image)[:,:,0]
Y = rgb2lab(1.0/255*image)[:,:,1:]
Y /= 128
# X = X.reshape(1, color_x, color_y, 1)
X = X.reshape(1, 360, 360, 1)
print("X =",X)
# Y = Y.reshape(1, color_x, color_y, 2)
Y = Y.reshape(1, 360, 360, 2)
print("Y =",Y)


# Neural network
model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

model.compile(optimizer='rmsprop', loss='mse')

model.fit(x=X, y=Y, batch_size=1, epochs=1000)
print(model.evaluate(X, Y, batch_size=1))

# csv2image("input.csv",color_x,color_y)
image = img_to_array(load_img('input.png'))
image = cv2.resize(image, (360, 360), interpolation=cv2.INTER_CUBIC)
image = np.array(image, dtype=float)
X = rgb2lab(1.0/255*image)[:,:,0]
X = X.reshape(1, 360, 360, 1)

output = model.predict(X)
output *= 128

# cur = np.zeros((color_x,color_y, 3))
cur = np.zeros((360,360,3))
cur[:,:,0] = X[0][:,:,0]
cur[:,:,1:] = output[0]
cur = cv2.resize(cur, (color_x, color_y), interpolation=cv2.INTER_CUBIC)

imsave("img_result.png", lab2rgb(cur))
imsave("img_gray_version.png", rgb2gray(lab2rgb(cur)))

