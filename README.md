# 🍃 Tea Disease Classification with TensorFlow

This project focuses on detecting and classifying diseases in tea leaves using deep learning. The model is trained using TensorFlow and performs image classification on a custom dataset of tea leaf images labeled with disease categories.

## 📁 Project Structure

* `training-checkpoint.ipynb`: Jupyter notebook containing the full pipeline of data preprocessing, model building, training, evaluation, and saving.
* `data/`: Directory containing the tea leaf image dataset structured in class-wise subdirectories.

## 🚀 Features

* Image classification using Convolutional Neural Networks (CNNs)
* Dataset loading with `image_dataset_from_directory`
* Data visualization and augmentation
* Model building using `tf.keras.Sequential`
* Model training and evaluation
* Confusion matrix plotting
* Model saving in `.h5` format

## 🛠️ Dependencies

To run the notebook, you need the following packages installed:

```bash
pip install tensorflow matplotlib numpy scikit-learn
```

## 🧪 Dataset Structure

The dataset should follow this structure:

```
data/
├── Healthy/
│   ├── image1.jpg
├── Disease1/
│   ├── image1.jpg
├── Disease2/
│   ├── image1.jpg
```

## 📊 Model Summary

* **Architecture**: Sequential CNN with `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense` layers
* **Loss Function**: Sparse Categorical Crossentropy
* **Optimizer**: Adam
* **Metrics**: Accuracy
* **Batch Size**: 32
* **Epochs**: 10 (modifiable)
* **Image Size**: 256x256

## 📈 Evaluation

* Accuracy metrics are visualized with `matplotlib`
* Confusion matrix is generated using `sklearn.metrics`
* Model is tested on a validation split from the dataset

## 💾 Model Saving

The trained model is saved using:

```python
model.save("tea_disease_model.h5")
```

You can later load this model using:

```python
from tensorflow.keras.models import load_model
model = load_model("tea_disease_model.h5")
```

## 📌 How to Run

1. Clone the repository or download the notebook.
2. Ensure your dataset is prepared in the correct folder structure.
3. Open `training-checkpoint.ipynb` in Jupyter or Colab.
4. Run all cells sequentially.

## 📚 References

* TensorFlow documentation: [https://www.tensorflow.org/api\_docs](https://www.tensorflow.org/api_docs)
* Dataset source: (Add your dataset source here)
* Image classification tutorial inspiration from TensorFlow Examples.


