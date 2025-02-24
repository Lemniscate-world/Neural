import tensorflow as tf

model = tf.keras.Sequential(name='MathTransformer', layers=[
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer='Adam')
