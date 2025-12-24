import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

path_dir1 = './LSW/'
path_dir2 = './KUM/'
path_dir3 = './JSJ/'

file_list1 = os.listdir(path_dir1)  # path에 존재하는 파일 목록 가져오기
file_list2 = os.listdir(path_dir2)
file_list3 = os.listdir(path_dir3)

file_list1_num = len(file_list1)
file_list2_num = len(file_list2)
file_list3_num = len(file_list3)


file_num = file_list1_num + file_list2_num + file_list3_num

# %% train용 이미지 준비
num = 0;
all_img = np.float32(np.zeros((file_num, 224, 224, 3)))  # 394+413+461
all_label = np.float64(np.zeros((file_num, 1)))

for img_name in file_list1:
    img_path = path_dir1 + img_name
    img = load_img(img_path, target_size=(224, 224))

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    all_img[num, :, :, :] = x

    all_label[num] = 0
    num = num + 1

for img_name in file_list2:
    img_path = path_dir2 + img_name
    img = load_img(img_path, target_size=(224, 224))

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    all_img[num, :, :, :] = x

    all_label[num] = 1
    num = num + 1

for img_name in file_list3:
    img_path = path_dir3 + img_name
    img = load_img(img_path, target_size=(224, 224))

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    all_img[num, :, :, :] = x

    all_label[num] = 2
    num = num + 1

# 이미지 섞기

n_elem = all_label.shape[0]
indices = np.random.choice(n_elem, size=n_elem, replace=False)

all_label = all_label[indices]
all_img = all_img[indices]

#훈련셋 테스트셋 분할
num_train = int(np.round(all_label.shape[0] * 0.8))
num_test = int(np.round(all_label.shape[0] * 0.2))

train_img = all_img[0:num_train, :, :, :]
test_img = all_img[num_train:, :, :, :]

train_label = all_label[0:num_train]
test_label = all_label[num_train:]


# %%
# create the base pre-trained model
IMG_SHAPE = (224, 224, 3)

base_model = ResNet50(input_shape=IMG_SHAPE, weights='imagenet', include_top=False)
base_model.trainable = False
base_model.summary()
print("Number of layers in the base model: ", len(base_model.layers))

GAP_layer = GlobalAveragePooling2D()
dense_layer = Dense(3, activation=tf.nn.softmax)


model = Sequential([
    base_model,
    GAP_layer,
    dense_layer
])

base_learning_rate = 0.01
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

model.fit(train_img, train_label, epochs=10)

# save model
model.save("")

print("Saved model to disk")