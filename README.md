# Anime Face Generation GAN

## Overview
This project implements a Generative Adversarial Network (GAN) to generate anime-style faces from random noise. The model is trained using the Kaggle Anime Face Dataset, leveraging deep learning techniques to enhance image diversity, resolution, and authenticity.

## Dataset
- **Source:** Kaggle Anime Face Dataset
- **Directory Structure:**
  - `./animefacedataset/images/` contains anime face images.

## Dependencies
Ensure you have the following libraries installed:

```bash
pip install torch torchvision matplotlib tqdm
```

## Data Preprocessing
- Images are loaded using `torchvision.datasets.ImageFolder`.
- Transformations applied:
  - Resizing to 64x64
  - Center Cropping
  - Normalization (scaling range from [-1,1])

## Model Architecture

### Discriminator
- A convolutional neural network that classifies images as real or fake.
- Uses LeakyReLU activations and batch normalization.
- Outputs a single probability score.

### Generator
- A transposed convolutional neural network that generates anime faces from random noise.
- Uses ReLU activations and batch normalization.
- Outputs a 64x64 RGB image with a `Tanh` activation.

## Training Process

### Discriminator Training
- Classifies real and fake images.
- Uses binary cross-entropy loss.
- Updates weights based on classification accuracy.

### Generator Training
- Generates fake images to fool the discriminator.
- Uses binary cross-entropy loss.
- Updates weights based on the discriminator's feedback.

## Training Execution
Run the training process:

```python
lr = 0.0002
epochs = 25
history = fit(epochs, lr)
```

## Saving Model & Generated Images
Saves the trained models:

```python
torch.save(generator.state_dict(), 'G.ckpt')
torch.save(discriminator.state_dict(), 'D.ckpt')
```

Saves generated images after every epoch.

## Sample Output
Display generated images:

```python
from IPython.display import Image
Image('./generated/generated-images-0001.png')
```

## Results
- The generator progressively improves in generating realistic anime faces.
- Loss values and discriminator scores help monitor training performance.

## Future Improvements
- Enhance model architecture for better quality images.
- Train on larger datasets to improve diversity.
- Implement progressive growing GANs for higher resolution output.
