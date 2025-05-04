# Solving Mathematical Equations Using PINN

This repository is dedicated to the diploma project titled **"Solving Mathematical Equations Using the Physics-Informed Neural Networks (PINN) Method"**. The project explores the application of PINN for solving mathematical equations, with a focus on testing various architectures of PINN on the following problems:

## Key Objectives

1. **1D and Multi-Dimensional Burgers' Equation**:
   - The repository includes experiments to test and compare different PINN architectures for solving the one-dimensional and multi-dimensional Burgers' equation.

2. **KdV-Burgers Equation**:
   - After validating the PINN approach on the Burgers' equation, the method will be applied to solve the Korteweg-de Vries-Burgers (KdV-Burgers) equation.

## Repository Structure

- **`notebooks/`**: Contains Jupyter notebooks for running experiments and visualizing results.
- **`models/`**: Includes implementations of various PINN architectures.
- **`tasks/`**: Defines the mathematical equations and boundary conditions.
- **`data/`**: Stores results and loss data for different experiments.
- **`common/`**: Utility functions for training and plotting.
- **`pics/`**: Visualizations and plots generated during experiments.

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebooks in the `notebooks/` directory to reproduce the experiments.

## Requirements

- Python 3.11+
- PyTorch
- NumPy
- Matplotlib

## Author
This project is part of a diploma thesis and is maintained by Margarita Kikot.