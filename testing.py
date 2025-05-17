import torch
from torch import nn
import pandas as pd
import json

def main():

    df = pd.read_csv("test.csv") 
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    passenger_ids = df["PassengerId"]

    df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
    df = pd.get_dummies(df)
    
    with open("input_columns.json", "r") as f:
        expected_columns = json.load(f)

    df = df.reindex(columns=expected_columns, fill_value=0)
    df = df.astype('float32')

    # Recreate model architecture
    model = nn.Sequential(
        nn.Linear(df.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 2)
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} device")

    model.load_state_dict(torch.load("titanic_model.pt"))
    model.eval()
    model.to(device=device)

    x = torch.tensor(df.values, dtype=torch.float32)
    x = x.to(device)

    with torch.no_grad():
        outputs = model(x) 
        preds = torch.argmax(outputs, dim=1)

    result_df = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": preds.cpu().numpy()
    })
    result_df.to_csv("predictions.csv", index=False)

if __name__ == "__main__":
    main()