import tensorflow as tf

class TransformerEncoder(tf.keras.layers.Layer):
                # ... (keep previous implementation)
            
            class TransformerDecoder(tf.keras.layers.Layer):
                # ... (keep previous implementation)

# Model construction using Functional API
inputs = tf.keras.Input(shape=(512,))
x = inputs
x = TransformerEncoder(num_heads=8, ff_dim=2048, dropout=0.1)(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2
)
