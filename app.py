import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, Model # Import specific layers used in your model
import numpy as np
from PIL import Image # For image manipulation
import io # To handle image bytes

# --- Configuration ---
NOISE_DIM = 100 # Must match the noise dimension used in training
NUM_CLASSES = 10 # Digits 0-9

# --- Model Architecture (Must match Generator from training script) ---
# Re-define the Generator model architecture exactly as it was in your training script.
# This is crucial for loading weights correctly if you're loading weights directly,
# or for understanding the expected input/output if loading a full model.
def make_generator_model():
    """Defines the Generator model architecture."""
    noise_input = layers.Input(shape=(NOISE_DIM,))
    label_input = layers.Input(shape=(NUM_CLASSES,))

    noise_dense = layers.Dense(7 * 7 * 128)(noise_input)
    noise_reshaped = layers.Reshape((7, 7, 128))(noise_dense)

    label_dense = layers.Dense(7 * 7 * 1)(label_input)
    label_reshaped = layers.Reshape((7, 7, 1))(label_dense)

    merged_input = layers.Concatenate()([noise_reshaped, label_reshaped])

    x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(merged_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    output_image = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)

    model = Model(inputs=[noise_input, label_input], outputs=output_image, name='Generator')
    return model

# --- Load the Trained Generator Model ---
@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_trained_model():
    """Loads the trained generator model weights."""
    model_path = 'generator.h5' 

    try:
        # Option 1: Load a full Keras model (if saved with model.save())
        # This is generally preferred if the pre-trained model was saved as a full model
        model = tf.keras.models.load_model(model_path)
        st.success(f"Generator model loaded successfully from {model_path}!")
        return model
    except Exception as e_full_load:
        st.warning(f"Could not load full model: {e_full_load}. Attempting to load weights into pre-defined architecture.")
        # Option 2: If Option 1 fails, load weights into the predefined architecture
        # This assumes your 'make_generator_model' exactly matches the pre-trained model's architecture.
        try:
            model = make_generator_model() # Create an instance of the model
            model.load_weights(model_path) # Load only the weights
            st.success(f"Generator weights loaded successfully into defined architecture from {model_path}!")
            return model
        except Exception as e_weights_load:
            st.error(f"Error loading model weights into architecture: {e_weights_load}")
            st.warning(f"Please ensure '{model_path}' is in the same directory as 'app.py' and is a valid Keras model/weights file for the specified architecture.")
            return None # Return None if model loading completely fails

# Load the model when the app starts
generator_model = load_trained_model()

# --- Streamlit Web Application Interface ---
st.set_page_config(layout="wide", page_title="Handwritten Digit Generator")

st.title("✍️ Handwritten Digit Generator (0-9)")
st.write("Select a digit below, and the app will generate 5 unique handwritten images of that digit using a trained AI model (CGAN).")

# User input for digit selection
selected_digit = st.slider("Choose a digit (0-9):", 0, 9, 0)

# Button to trigger image generation
if st.button("Generate Images"):
    if generator_model:
        st.subheader(f"Generated Images for Digit: {selected_digit}")
        
        # Generate 5 unique images
        generated_images = []
        for _ in range(5):
            # Generate random noise for diversity
            noise = tf.random.normal([1, NOISE_DIM])
            # One-hot encode the selected digit
            label = tf.keras.utils.to_categorical([selected_digit], NUM_CLASSES)
            
            # Predict (generate) image
            image = generator_model([noise, label], training=False)[0] # Get the first (and only) image from the batch
            
            # Denormalize the image from [-1, 1] to [0, 255] for display
            image = (image * 0.5 + 0.5) * 255
            image = image.numpy().astype(np.uint8) # Convert to NumPy array and then to uint8
            image = image.reshape(28, 28) # Reshape to 28x28 for display
            
            # Convert NumPy array to PIL Image and then to bytes for Streamlit
            img_pil = Image.fromarray(image, mode='L') # 'L' for grayscale
            buf = io.BytesIO()
            img_pil.save(buf, format="PNG")
            generated_images.append(buf.getvalue())
        
        # Display images in a grid format
        cols = st.columns(5) # Create 5 columns
        for i, img_bytes in enumerate(generated_images):
            with cols[i]:
                st.image(img_bytes, caption=f"Image {i+1}", use_column_width=True)
                
        st.success("Images generated! You can select another digit or generate more.")
    else:
        st.error("Model not loaded. Cannot generate images.")

st.markdown("---")
st.markdown("Powered by TensorFlow/Keras and Streamlit")

