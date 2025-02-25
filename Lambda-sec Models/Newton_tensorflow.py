import tensorflow as tf
from tensorflow.keras import layers

class TransformerEncoder(layers.Layer):
    def __init__(self, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=ff_dim//num_heads  # Proper dimension calculation
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(ff_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
    
    def call(self, inputs):
        attn_output = self.attn(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

# Model construction using Functional API
inputs = tf.keras.Input(shape=(None, 512))
x = inputs
x = TransformerEncoder(num_heads=8, ff_dim=2048, dropout=0.1)(x)
x = layers.Dense(512, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

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
    validation_split=0.2,
    verbose=1
)
