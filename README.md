# Learning-Probability-Density-Functions-using-data-only
# GAN-Based Air Quality Data Synthesis

A Generative Adversarial Network (GAN) implementation for synthesizing realistic NO2 (Nitrogen Dioxide) air quality data using PyTorch.

## Overview

This project implements a GAN architecture to learn and generate synthetic NO2 pollution data distributions. The model transforms real NO2 measurements through a custom mathematical transformation and trains a generator to produce realistic synthetic samples that match the distribution of the transformed data.

## Features

- **Data Processing**: Loads and preprocesses NO2 air quality data from CSV
- **Custom Transformation**: Applies a unique mathematical transformation based on roll number parameters
- **GAN Architecture**: 
  - Generator: 3-layer neural network with ReLU activations
  - Discriminator: 3-layer neural network with LeakyReLU activations
- **Training**: 5000 epochs with Adam optimizer
- **Visualization**: Probability density comparison between real and synthetic data

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
torch
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/gan-air-quality-synthesis.git
cd gan-air-quality-synthesis
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch
```

## Usage

1. Ensure you have a CSV file named `Data_1.csv` with a column named `no2` containing NO2 measurements

2. Run the script:
```bash
python assignment_4_predictive.py
```

3. The script will:
   - Load and preprocess the NO2 data
   - Apply the custom transformation
   - Train the GAN for 5000 epochs
   - Generate synthetic samples
   - Display a probability density comparison plot

## Model Architecture

### Generator
- Input: Random noise (1 dimension)
- Hidden layers: 32 neurons each with ReLU activation
- Output: Synthetic data sample (1 dimension)

### Discriminator
- Input: Real or fake data sample (1 dimension)
- Hidden layers: 32 neurons each with LeakyReLU activation
- Output: Probability score (0-1) via Sigmoid

## Transformation Formula

The data transformation is based on the roll number `102317054`:

```
α = 0.5 × (roll_number % 7)
β = 0.3 × ((roll_number % 5) + 1)
z = x_scaled + α × sin(β × x_scaled)
```

Where `x_scaled` is the standardized NO2 data.

## Training Details

- **Epochs**: 5000
- **Batch Size**: 128
- **Optimizer**: Adam (learning rate: 0.0002)
- **Loss Function**: Binary Cross-Entropy (BCE)
- **Synthetic Samples Generated**: 8000

## Output

The script produces a visualization comparing:
- **Actual Transformed Data (z)**: Original NO2 data after transformation
- **GAN Synthesized Data (ẑ)**: Generated synthetic data

The plot shows probability density functions to evaluate how well the GAN has learned the data distribution.

## File Structure

```
.
├── assignment_4_predictive.py    # Main script
├── Data_1.csv                     # Input data (not included)
└── README.md                      # This file
```

## Results

The trained GAN generates synthetic NO2 data that closely matches the probability distribution of the transformed real data, as visualized in the density plot comparison.

## Notes

- The script uses `latin1` encoding for CSV reading
- Bad lines in the CSV are automatically skipped during loading
- Random sampling ensures different batches in each epoch
- The generator is set to evaluation mode before generating final synthetic samples

## License

This project is available for educational and research purposes.

## Author

Roll Number: 102317054

## Acknowledgments

- Built with PyTorch
- Visualization using Matplotlib and Seaborn
- Data preprocessing with scikit-learn
