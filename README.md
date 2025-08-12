# Fashion-MNIST Classification (Keras, TensorFlow)

## 1) Problem Statement and Goal of Project

Build a supervised image classifier that maps 28×28 grayscale images of clothing items to one of **10 classes** using a neural network. The goal is to implement a clean baseline, train it end-to-end, and evaluate generalization on the official test split.

## 2) Solution Approach

* **Data**: `keras.datasets.fashion_mnist` (loaded as `(x_train, y_train), (x_test, y_test)`).
* **Preprocessing**: Pixel intensity normalization to `[0, 1]` via `x_train, x_test = x_train/255.0, x_test/255.0`.
* **Model**: Multilayer Perceptron (MLP) built with `keras.Sequential`

  ```
  Flatten(input_shape=[28,28])
  Dense(100, activation="relu")
  Dense(75,  activation="relu")
  Dense(10,  activation="softmax")
  ```
* **Training**:

  * Loss: `sparse_categorical_crossentropy`
  * Optimizer: `SGD` (default params as in Keras; no custom LR specified in the notebook)
  * Metrics: `accuracy`
  * Epochs: **50**
  * Validation: `validation_split=0.15` from the training set
* **Model Introspection**: Accessed `model.layers`, retrieved weights/biases of the first Dense layer, and visualized learning curves (loss/accuracy) using Matplotlib.

## 3) Technologies & Libraries

* **TensorFlow/Keras**: 2.10.0 (verified in notebook output).
* **Python**: 3.9.21 (notebook metadata).
* **NumPy**, **Matplotlib** for data handling and plotting.

## 4) Description about Dataset

* **Source**: `keras.datasets.fashion_mnist`
* **Shapes observed in notebook**:

  * `x_train.shape → (60000, 28, 28)`
  * `y_train.shape → (60000,)`
* **Normalization**: Inputs scaled to `[0,1]` by dividing by 255.
* **Split**: Predefined train/test from Keras; additional **15%** of training data held out for validation in `model.fit`.

## 5) Installation & Execution Guide

**Prerequisites**

```bash
pip install tensorflow==2.10.* numpy matplotlib
```

**Run**

* Open `ch01_calssification.ipynb` and execute cells top-to-bottom.

## 6) Key Results / Performance

* **Test set**:
  `loss: 0.3555`    `accuracy: 0.8761`
  (from the `model.evaluate(x_test, y_test)` output captured in the notebook)

* **Training dynamics**: The notebook plots training vs. validation **loss** and **accuracy** across 50 epochs using Matplotlib (`history.history[...]`).

## 7) Screenshots / Sample Output

*(No external screenshots included; below are direct snippets from the notebook outputs.)*

* **Sanity check (3 samples)**

  ```
  np.argmax(model.predict(x3).round(3), axis=1) → [2, 1, 1]
  y_test[1:4]                                → [2, 1, 1]
  ```
* **Learning curves**: A Matplotlib figure showing `train/val loss` and `train/val accuracy` over 50 epochs is generated in-notebook.

## 8) Additional Learnings / Reflections

* Practiced two ways of defining Keras models (documented in Persian markdown cells).
* Performed **model introspection** by inspecting `model.layers` and fetching weights/biases for early layers to understand parameter shapes and initialization.
* Established a solid **baseline MLP** workflow on Fashion-MNIST with clear preprocessing, training, validation, and evaluation—useful for comparing against future CNN or regularization variants. *(More advanced variants are not provided in this notebook.)*

---

## 👤 Author

**Mehran Asgari**
**Email:** [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com)
**GitHub:** [https://github.com/imehranasgari](https://github.com/imehranasgari)

---

## 📄 License

This project is licensed under the MIT License – see the `LICENSE` file for details.

---

> 💡 *Some interactive outputs (e.g., plots, widgets) may not display correctly on GitHub. If so, please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering.*
