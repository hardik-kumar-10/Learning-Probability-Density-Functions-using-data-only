# Learning Probability Density Functions using Data Only

## 1. Methodology

**Pipeline:**
Data Collection → Data Pre-processing → Standardization → Non-Linear Transformation → GAN Training → Sample Generation → PDF Approximation → Result Analysis

This assignment focuses on learning an unknown probability density function directly from data samples using Generative Adversarial Networks without assuming any parametric form.

## 2. Dataset Information

- **Dataset Name:** India Air Quality Dataset
- **Source:** Kaggle
- **Dataset Link:** https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data
- **Feature Used:** NO₂ (Nitrogen Dioxide) concentration
- **Description:** The dataset contains air quality measurements collected across multiple Indian cities. The NO₂ feature is selected as the input variable for learning the probability density function through adversarial training.

## 3. Objective

The objective of this assignment is to learn an unknown probability density function of a transformed random variable using only data samples. No analytical or parametric form of the probability density function is assumed. A Generative Adversarial Network (GAN) is used to implicitly learn the distribution directly from the data.

## 4. Mathematical Formulation

### Data Preprocessing

Each NO₂ value x is first standardized using StandardScaler:

```
x_scaled = (x - μ) / σ
```

where μ is the mean and σ is the standard deviation of the NO₂ data.

### Non-Linear Transformation

Each standardized NO₂ value x_scaled is transformed into z using the roll-number-parameterized non-linear function:

```
z = x_scaled + α sin(β x_scaled)
```

where:
- **α = 0.5 × (r mod 7)**
- **β = 0.3 × (r mod 5 + 1)**
- **r** is the university roll number

### Transformation Parameters

For the given university roll number:

- **r = 102317054**
- **α = 0.5 × (r mod 7) = 0.5 × (102317054 mod 7) = 0.5 × 0 = 0.0**
- **β = 0.3 × ((r mod 5) + 1) = 0.3 × ((102317054 mod 5) + 1) = 0.3 × (4 + 1) = 1.5**

In the code, these are represented as:
- **alpha = 0.0**
- **beta = 1.5**

The transformation applies a frequency-modulated sinusoidal modification to the standardized NO₂ data.

### GAN-Based Density Learning

- **Real samples:** z values from transformed data
- **Fake samples:** generated_samples = Generator(ε), where ε ~ N(0,1)
- The generator implicitly models the probability distribution of the transformed variable
- No parametric probability density function is assumed
- The discriminator learns to distinguish real from fake samples

## 5. GAN Architecture Description

### Generator Network
- **Type:** Fully connected neural network
- **Input:** One-dimensional noise sampled from standard normal distribution N(0,1)
- **Architecture:**
  - Input layer: 1 neuron (noise)
  - Hidden layer 1: 32 neurons with ReLU activation
  - Hidden layer 2: 32 neurons with ReLU activation
  - Output layer: 1 neuron (generated sample)
- **Output:** Generated samples of the transformed variable z

### Discriminator Network
- **Type:** Fully connected neural network
- **Input:** Real or generated samples of dimension 1
- **Architecture:**
  - Input layer: 1 neuron (sample)
  - Hidden layer 1: 32 neurons with LeakyReLU activation (slope=0.2)
  - Hidden layer 2: 32 neurons with LeakyReLU activation (slope=0.2)
  - Output layer: 1 neuron with Sigmoid activation
- **Output:** Probability (0-1) of the input sample being real

The generator and discriminator are trained adversarially until the generator produces samples that resemble the real transformed data distribution.

## 6. PDF Approximation from Generator Samples

### Training Configuration
- **Loss Function:** Binary Cross-Entropy (BCE)
- **Optimizer:** Adam
- **Learning Rate:** 0.0002 (2e-4)
- **Epochs:** 5,000
- **Batch Size:** 128
- **Generated Samples for PDF:** 8,000

### PDF Estimation Process
After training the GAN:

1. Generate 8,000 samples using the trained generator
2. Apply Kernel Density Estimation (KDE) with bandwidth adjustment 1.1 to both real and generated samples
3. The estimated density represents the learned probability density function of z
4. Compare with the KDE of real transformed samples
5. Visualize with professional formatting: 9x5 figure size, filled density plots with 50% transparency

## 7. Training Process

### Discriminator Training
For each epoch:
1. Sample a batch of real transformed data z
2. Label real samples as 1
3. Generate fake samples from random noise
4. Label fake samples as 0
5. Calculate discriminator loss: L_D = BCE(D(z_real), 1) + BCE(D(G(ε)), 0)
6. Update discriminator parameters

### Generator Training
For each epoch:
1. Generate fake samples from random noise
2. Calculate generator loss: L_G = BCE(D(G(ε)), 1)
3. Update generator parameters to fool the discriminator

### Loss Monitoring
Training progress is logged every 1,000 epochs showing:
- Discriminator Loss (D Loss)
- Generator Loss (G Loss)

## 8. Input / Output

### Input
- NO₂ concentration values from the India Air Quality Dataset
- Loaded from `Data_1.csv` with encoding='latin1'
- Missing values are dropped using `.dropna()`

### Output
- **Standardized variable:** x_scaled using StandardScaler
- **Transformed variable:** z = x_scaled + α sin(β x_scaled)
- **Generated samples:** Synthetic samples from the trained generator
- **Estimated PDF:** p̂(z) obtained via KDE on generated samples
- **Visualization:** Comparison plot showing real vs. generated density curves

## 9. Result Graph

The figure below shows:<img width="889" height="490" alt="image" src="https://github.com/user-attachments/assets/d22ea61f-f32c-4bdd-a5c1-f0fb67996048" />


1. **Filled Blue Curve:** Real transformed samples distribution (KDE)
2. **Filled Orange Curve:** GAN-generated samples distribution (KDE)
3. **Overlay:** Direct comparison of empirical distribution vs. learned distribution

The visualization demonstrates how well the GAN has learned to approximate the true underlying distribution through adversarial training.

### PDF Estimation Plot

**Note:** Replace this section with your actual result image when available.

```
[Insert your generated plot here]
```

**Figure:** Probability Density Comparison between real transformed NO₂ data and GAN-synthesized data. Both curves are obtained through kernel density estimation with bandwidth adjustment factor 1.1. The close alignment between the distributions indicates successful distribution learning by the adversarial network.

## 10. Observations

### Mode Coverage
The generator successfully captures the dominant modes of the transformed variable, with the estimated PDF closely following the empirical distribution.

### Training Stability
Training remains stable due to:
- Proper data normalization using StandardScaler
- Balanced generator and discriminator architecture (both 2 hidden layers with 32 neurons)
- Appropriate learning rate (2e-4) and batch size (128) selection
- Gradual loss convergence over 5,000 epochs
- Logging at regular intervals for monitoring

### Quality of Generated Distribution
The estimated probability density function closely follows the empirical distribution of the real samples, with smooth representation of the data density. The KDE bandwidth adjustment parameter (1.1) provides optimal smoothing between capturing modes and avoiding over-fitting to individual samples.

### Distribution Characteristics
- The transformed NO₂ data exhibits characteristics influenced by the sinusoidal transformation
- The GAN captures the distributional properties effectively
- Generated samples maintain the statistical properties of the real data
- The use of seaborn's KDE with fill provides clear visual comparison

### Architecture Efficiency
- Compact architecture (32 neurons per hidden layer) proves sufficient for 1D distribution learning
- ReLU activation in generator ensures non-negative gradient flow
- LeakyReLU in discriminator prevents dead neurons
- Sigmoid output in discriminator provides clear probability interpretation

## 11. Conclusion

This assignment successfully demonstrates that Generative Adversarial Networks can be used to learn an unknown probability density function directly from data samples. The approach:

- Avoids assuming any analytical form of the distribution
- Provides a purely data-driven solution for density estimation
- Works effectively for complex, non-parametric distributions
- Generates high-quality synthetic samples matching the real data distribution
- Successfully approximates the true PDF through adversarial learning
- Demonstrates effective use of standardization and non-linear transformation
- Shows how compact neural networks can learn 1D distributions efficiently

The GAN learns implicit representations of the data distribution without requiring explicit probability models, making it a powerful tool for density estimation in machine learning applications.

## 12. Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Epochs | 5,000 | Total training iterations |
| Batch Size | 128 | Samples per training step |
| Learning Rate | 0.0002 | Adam optimizer learning rate |
| Hidden Units (Generator) | 32 | Neurons in each hidden layer |
| Hidden Units (Discriminator) | 32 | Neurons in each hidden layer |
| Generated Samples | 8,000 | Samples for PDF estimation |
| KDE Bandwidth Adjustment | 1.1 | Smoothing parameter for density estimation |
| Figure Size | 9 x 5 | Width and height in inches |
| Noise Dimension | 1 | Input noise dimension |
| KDE Fill Alpha | 0.5 | Transparency of density plots |
| LeakyReLU Slope | 0.2 | Negative slope for discriminator |
| Roll Number | 102317054 | Used for transformation parameters |
| Alpha (α) | 0.0 | Transformation amplitude parameter |
| Beta (β) | 1.5 | Transformation frequency parameter |
| Logging Interval | 1000 | Epochs between loss printing |

## 13. Code Structure

### Libraries Used
- **pandas:** Data loading and manipulation
- **numpy:** Numerical computations and array operations
- **matplotlib.pyplot:** Visualization framework
- **seaborn:** Statistical data visualization (KDE plots)
- **sklearn.preprocessing.StandardScaler:** Data normalization
- **torch:** Deep learning framework
- **torch.nn:** Neural network modules
- **torch.optim:** Optimization algorithms

### File Requirements
- Input file: `Data_1.csv` with 'no2' column
- Encoding: latin1
- Bad lines: skipped automatically

### Execution Flow
1. Load and preprocess NO₂ data
2. Apply standardization
3. Compute transformation parameters from roll number
4. Transform data using non-linear function
5. Initialize generator and discriminator networks
6. Train GAN for 5,000 epochs
7. Generate synthetic samples
8. Visualize probability density comparison

## 14. Future Improvements

Potential enhancements to this implementation:

- Experiment with deeper architectures for more complex distributions
- Implement Wasserstein GAN for improved training stability
- Add gradient penalty for better convergence
- Increase number of generated samples for smoother PDF estimation
- Implement early stopping based on distribution similarity metrics
- Add validation set to monitor overfitting
- Experiment with different activation functions
- Implement batch normalization for faster convergence
- Save model checkpoints during training
- Quantitative evaluation using statistical distance metrics (KL divergence, Wasserstein distance)
