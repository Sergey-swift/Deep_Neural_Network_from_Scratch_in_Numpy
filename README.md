```markdown
# DNN from Scratch in Numpy Portfolio Project: Fashion-MNIST Classification

## Project Description
Hey there! Welcome to my DNN from Scratch in Numpy project built entirely from scratch using NumPy. This project is a hands-on demonstration of how I implement DNN fundamentals—from forward propagation and dropout to L2 regularization and Adam optimization. I applied this model to the Fashion-MNIST dataset to classify fashion items, showcasing my journey from concept to code.

## Dataset
I'm using the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist), which consists of 60,000 training images and 10,000 test images of clothing items (think sneakers, shirts, bags, etc.). Each image is a 28x28 grayscale picture that packs a lot of style into a small frame.

---

## Tools and Libraries
- Python
- NumPy
- Matplotlib
- TensorFlow (for data loading)
- scikit-learn
- seaborn

---

## Key Features
1. **Modular Implementation:**
   - **Forward Propagation:** Utilizes ReLU, Sigmoid, and Softmax activations.
   - **Backward Propagation:** Includes dropout and L2 regularization for robust learning.
   - **Adam Optimization:** Employs bias-corrected first and second moment estimates for smooth training.
2. **Mini-Batch Gradient Descent:**
   - Generates random mini-batches with a reproducible seed.
   - Implements learning rate decay to fine-tune the training process over epochs.
3. **Visualization Utilities:**
   - Displays sample training examples.
   - Plots confusion matrices and prediction grids.
   - Visualizes first-layer filters and hidden layer activations.
4. **Evaluation:**
   - Calculates and displays training and test accuracies.
   - Provides detailed error analysis with a confusion matrix.
5. **Flexible and Modular Structure:**
   - The current modular design allows you to easily split the code into separate files or modify the architecture with only minor changes to the main function.
   - Future improvements can include deeper networks, alternative architectures, or hyperparameter fine-tuning with minimal refactoring.

---

## Current Model Performance
- **Train Accuracy:** 90.30%
- **Test Accuracy:** 87.03%

*Note: These metrics are based on a modest training regime. With longer training and further hyperparameter fine-tuning, the model can achieve even better results.*

---

## How to Use
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Sergey-swift/Deep_Neural_Network_from_Scratch_in_Numpy.git
   ```
2. **Navigate to the Project Directory:**
   ```bash
   cd Deep_Neural_Network_from_Scratch_in_Numpy
   ```
3. **Install the Required Packages:**
   ```bash
   pip install numpy matplotlib tensorflow scikit-learn seaborn
   ```
4. **Run the Project Script:**
   ```bash
   python Deep_Neural_Network_from_Scratch_in_Numpy.py
   ```

---

## Results and Visualizations
After training the model, you'll see:
- **Accuracy Metrics:** Training and test accuracy scores to gauge performance.
- **Confusion Matrix:** A detailed view of how the model performs on the test set.
- **Visual Insights:** Sample grids of training examples, prediction results, first-layer filters, hidden layer activations, and even misclassified instances for extra insight.

---

## Repository Structure
```
Deep_Neural_Network_from_Scratch_in_Numpy/
├── Deep_Neural_Network_from_Scratch_in_Numpy.py   # Main implementation of the DNN
├── README.md                  # Project documentation (this file)
└── fashion_mnist_model.npz  # Saved model parameters
```

---

## Author
Sergey Swift   
[GitHub Profile](https://github.com/Sergey-swift)

Thanks for checking out my project—hope you enjoy exploring it as much as I enjoyed building it!
```