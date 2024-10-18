# CS5787-DeepLearning-Assignment3

# Image Generation with GANs

This project implements three types of Generative Adversarial Networks (GANs) to generate clothing images (T-shirts and Dresses) using the Fashion MNIST dataset. The GAN models included in this project are:
- Deep Convolutional GAN (DCGAN)
- Wasserstein GAN (WGAN)
- Wasserstein GAN with Gradient Penalty (WGAN-GP)

## Requirements

Make sure you have the following libraries installed:

```bash
pip install torch torchvision matplotlib
```

## Generating New Images
After training the models, you can generate new images using the trained weights. Here’s how to do it:

Load the Trained Model:

Ensure the model weights are saved in the models directory. Adjust the model_path variable in main.py to point to the appropriate weights for each GAN type.
Generate Images:

Use the following code snippet to generate images with the trained models:

```

import torch
from torchvision.utils import save_image

def generate_images(generator, num_images, latent_dim, device='cpu'):
    generator = generator.to(device)
    z = torch.randn(num_images, latent_dim).to(device)
    with torch.no_grad():
        generated_images = generator(z)
    return generated_images

# Example usage
device = 'cuda' if torch.cuda.is_available() else 'cpu'
latent_dim = 100
num_images = 6  # Adjust the number of images to generate

# Load your trained models here
dcgan_gen = Architecture_2_Generator() # Load your trained DCGAN generator
wgan_gen = Architecture_2_Generator()  # Load your trained WGAN generator
wgan_gp_gen = Architecture_2_Generator() # Load your trained WGAN-GP generator

# Generate images
dcgan_images = generate_images(dcgan_gen, num_images, latent_dim, device)
wgan_images = generate_images(wgan_gen, num_images, latent_dim, device)
wgan_gp_images = generate_images(wgan_gp_gen, num_images, latent_dim, device)

# Save the generated images
for i, img in enumerate(dcgan_images):
    save_image(img, f'generated_images/dcgan_image_{i}.png')

```

