import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from PIL import Image

def load_image(image_path, max_dim=512):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = np.array(img)
    img = tf.image.resize(img, (max_dim, max_dim))
    img = img / 255.0
    return tf.expand_dims(img, axis=0)

def show_image(img):
    img = np.squeeze(img, axis=0)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Load pre-trained NST model
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def apply_style(content_path, style_path):
    content_image = load_image(content_path)
    style_image = load_image(style_path)
    
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    show_image(stylized_image)
    return stylized_image

# Example usage
if __name__ == "__main__":
    content_img_path = "s1.jpeg"  # Replace with actual content image path
    style_img_path = "ss.jpeg"  # Replace with actual style image path
    stylized_img = apply_style(content_img_path, style_img_path)
