import tensorflow as tf

model = tf.keras.Sequential(name='MyModel', layers=[
    tf.keras.layers.Flatten(input_shape=(None, 28, 28)),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(units=10, activation='softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer='Adam')
