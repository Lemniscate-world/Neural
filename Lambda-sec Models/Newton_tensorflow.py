import tensorflow as tf

model = tf.keras.Sequential(name='MathTransformer', layers=[
    # Transformer Encoder
                encoder_input = tf.keras.Input(shape=(512,))
                attention = tf.keras.layers.MultiHeadAttention(
                    num_heads=8, key_dim=2048//8)(encoder_input, encoder_input)
                x = tf.keras.layers.Dropout(0.1)(attention)
                x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + encoder_input)
                x = tf.keras.layers.Dense(2048, activation='relu')(x)
                x = tf.keras.layers.Dense(512)(x)
                x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
                model.add(tf.keras.Model(inputs=encoder_input, outputs=x))
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer='adam')
