from PIL import Image, ImageDraw, ImageFont
import os

# Prepare content
dsl_input = """network MyModel {
    input: (None, 28, 28)
    layers:
        Dense(128, activation="relu")
        Dropout(rate=0.2)
        Output(units=10, activation="softmax")
    loss: "categorical_crossentropy"
    optimizer: "Adam"
}"""

tf_code = """import tensorflow as tf

model = tf.keras.Sequential(name='MyModel', layers=[
    tf.keras.layers.Flatten(input_shape=(None, 28, 28)),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(units=10, activation='softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer='Adam')"""

# Settings
width, height = 800, 600
font_size = 20
bg_color = (255, 255, 255)  # White background
text_color = (0, 0, 0)      # Black text
font_path = "DejaVuSansMono.ttf"  # Use a monospaced font (adjust path if needed)

# Try to load a monospaced font; fallback to default if not found
try:
    font = ImageFont.truetype(font_path, font_size)
except:
    font = ImageFont.load_default()

# Function to create a frame
def create_text_frame(text, title=""):
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    if title:
        draw.text((20, 20), title, font=font, fill=text_color)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        draw.text((20, 50 + i * (font_size + 5)), line, font=font, fill=text_color)
    return img

# Create frames
frames = [
    create_text_frame(dsl_input, "DSL Input"),
    create_text_frame("Generating TensorFlow Code...", "Processing"),
    create_text_frame(tf_code, "Generated TensorFlow Code"),
    create_text_frame("Success! Ready to run in TensorFlow.", "Completed")
]

# Save as GIF
gif_path = "neural_code_generator.gif"
frames[0].save(
    gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=1500,  # 1.5 seconds per frame
    loop=0
)

print(f"GIF saved as {gif_path}")
