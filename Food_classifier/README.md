# Fruit and Vegetable Image Recognition

This repository is a **study project** focusing on **image classification** of fruits and vegetables. It uses a **pre-trained EfficientNet-B1** model fine-tuned on a dataset of various fruits and vegetables. The goal is to demonstrate how to **build, train, evaluate, and deploy** a PyTorch model for classification tasks.

## Project Structure

- **Model_Creation.ipynb**  
  - Jupyter notebook for **training and evaluation** (found that 6 epochs is optimal to avoid overfitting).  
- **engine.py**  
  - Contains **training**, **validation**, **testing** loops and utility functions for logging/clearing logs.  
- **model.py**  
  - Defines functions to **create**, **load**, and **predict** with the model.  
- **demo/fruits_vegs_mini/app.py**  
  - **Gradio** web app to upload images and get real-time predictions from the trained model.  
- **requirements.txt**  
  - Lists all the Python dependencies needed to run this project.

## Setup Instructions

1. **Clone the Repository**  
   ```bash
   git clone <repository-url>
   cd Food_classifier
   ```

2. **Install Dependencies**  
   Make sure you have Python installed. Then:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Dataset**  
   - The dataset is automatically downloaded from Kaggle using the **kagglehub** library.  
   - Ensure you have **Kaggle API credentials** set up for automatic download.

4. **Run the Jupyter Notebook**  
   - Open `Model_Creation.ipynb` in Jupyter (or VS Code, Google Colab, etc.) and run the cells to:
     - **Train** the model for a specified number of epochs (6 is recommended to avoid overfitting).
     - **Evaluate** on validation/test sets.
     - **Visualize** metrics (loss, accuracy, confusion matrix).

5. **Launch the Gradio App**  
   - You can interact with the trained model via a **web interface**:
     ```bash
     python demo/fruits_vegs_mini/app.py
     ```
   - A local URL will appear, letting you **upload images** and see classification results.

## Model Details

- **Architecture**: [EfficientNet-B1](https://pytorch.org/vision/main/models/efficientnet.html) (pre-trained on ImageNet).  
- **Training**: 
  - Fine-tuned for up to 10 epochs (but found that **6 epochs** was sufficient to prevent overfitting).  
  - Achieves **~96% accuracy** on the test set (depending on exact hyperparameters).  
- **Known Data Quirks**:  
  - **Bell Pepper** vs. **Capsicum** and **Corn** vs. **Sweetcorn** might cause confusion (they’re essentially the same item).  
  - **Potatoes** class has limited training samples, so performance may be weaker.  
  - This is primarily a **learning project**, so some dataset labels may be imperfect.

## Usage

- **Training**:  
  - Inside the notebook, adjust hyperparameters (e.g., learning rate, epochs) and run training cells.  
- **Prediction**:  
  - Use the **Gradio app** (`app.py`) to upload an image and see top class predictions.  
  - Or call `predict()` directly from `model.py` in your own scripts.



## Disclaimer

- This project was developed **for educational purposes**.  
- The dataset may contain **duplicate or mislabeled classes** (e.g., “Capsicum” vs. “Bell Pepper”).  
- **Not intended for production** usage; accuracy and robustness are limited by the dataset and scope of the study. The training and validation datasets contained a lot of junk images.

## Contact

For questions or suggestions, please reach out to **Artemii Pazhitnov / artpazhitnov@gmail.com** (replace with your actual info).
