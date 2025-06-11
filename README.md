# LoRA from Scratch on MNIST

This project demonstrates a from-scratch implementation of **LoRA (Low-Rank Adaptation)** in PyTorch, applied to a simple 3-layer Artificial Neural Network (ANN) on the MNIST dataset.

## What is LoRA?
LoRA is a parameter-efficient fine-tuning technique that injects trainable low-rank matrices into existing layers (typically linear layers) of a neural network, allowing you to adapt large models with a small number of additional parameters.

- **r**: The rank of the low-rank matrices (controls adapter size)
- **alpha**: Scaling factor for the LoRA update

## Project Structure
- `execution.ipynb`: Jupyter notebook with all code, experiments, and visualizations
- `README.md`: This file

## How to Run
1. Install dependencies (PyTorch, torchvision, matplotlib, numpy, etc.)
2. Open `execution.ipynb` in Jupyter or VSCode
3. Run all cells to:
    - Load MNIST
    - Define a 3-layer ANN
    - Implement and apply LoRA
    - Train and evaluate with different LoRA settings
    - Visualize results

## Experiments
- Compare LoRA models with different ranks (`r`) and scaling factors (`alpha`)
- Visualize test predictions
- Compare trainable parameter counts with a standard ANN

## Example Results
- LoRA can achieve good accuracy with far fewer trainable parameters than a standard ANN.

## References
- [LoRA: Low-Rank Adaptation of Large Language Models (arXiv:2106.09685)](https://arxiv.org/abs/2106.09685)
- [PyTorch Documentation](https://pytorch.org/)

---
Feel free to fork, modify, and experiment further! 