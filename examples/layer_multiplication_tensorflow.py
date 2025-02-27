import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

# Model construction using Functional API
# Input layer with shape (None, 32)
inputs = tf.keras.Input(shape=(None, 32))
x = inputs
# Dense layer with 64 units
x = layers.Dense(units=64, activation='relu')(x)
# Dropout layer with rate 0.5
x = layers.Dropout(rate=0.5)(x)
# Dense layer with 10 units
x = layers.Dense(units=10, activation='softmax')(x)

# Build the model
model = tf.keras.Model(inputs=inputs, outputs=outputs)
# Compile model with Adam optimizer and categorical_crossentropy loss
model.compile(loss='categorical_crossentropy', optimizer=Adam())
# Training configuration
model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
