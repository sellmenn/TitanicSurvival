# Titanic Survival Prediction with PyTorch

This project uses a simple PyTorch neural network to predict passenger survival on the Titanic dataset (`train.csv` and `test.csv`). The script handles training and prediction in one file — from data preprocessing to model saving and inference.

---

## 📄 What This Script Does

1. **Loads and preprocesses** `train.csv`
2. **Fills missing values** (median for Age, mode for Embarked)
3. Applies **one-hot encoding** on categorical variables
4. Splits into training and test sets
5. Trains a simple feedforward neural network
6. Saves the model (`titanic_model.pt`) and input column names (`input_columns.json`)
7. Loads and preprocesses `test.csv`, aligning it to training columns
8. Uses the trained model to predict survival
9. Saves predictions to `predictions.csv` with `PassengerId` and `Survived`

---

## 🧠 Features Used

- `Pclass`
- `Sex` (one-hot encoded)
- `Age` (filled using median)
- `SibSp`, `Parch`
- `Embarked` (filled using mode, one-hot encoded)

Dropped: `PassengerId`, `Name`, `Ticket`, `Fare`

---

## 🧪 Model Architecture

```python
nn.Sequential(
    nn.Linear(input_size, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 2)
)
