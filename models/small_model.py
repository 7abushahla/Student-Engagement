import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2

root_data_dir = "./Dataset"
img_width, img_height = 128, 128
num_classes = 3

#using functional API

# 1. Define the Input layer
input_layer = Input(shape=(img_width, img_height, 3))

# 2. Load the pre-trained MobileNetV2 model without the top classification layers
base_model = MobileNet(weights='imagenet', include_top=False, input_tensor=input_layer, input_shape=(img_width, img_height, 3), alpha=0.25)
base_model.trainable = True

# 4. Add custom layers on top of the base_model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x) 
x = Dropout(0.6)(x)
output_layer = Dense(num_classes, activation='softmax')(x)

# 5. Create the Functional API model
small_model = Model(inputs=input_layer, outputs=output_layer)

# 6. Compile the model with the SGD optimizer
opt = SGD(learning_rate=0.01)
small_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Optional: Display the model architecture
small_model.summary()