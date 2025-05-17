import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
import json

TEST_SIZE = 0.1

def main():

    class SurvivalData(Dataset):
        def __init__(self, data_file):
            df = pd.read_csv("train.csv") # Create df from csv
            df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
            df["Age"] = df["Age"].fillna(df["Age"].median())
            df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
            df = pd.get_dummies(df)
            df= df.astype('float32')
            self.labels = df["Survived"]
            self.data = df.drop("Survived", axis=1)

        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, index):
            data = torch.tensor(self.data.iloc[index], dtype=torch.float32)
            label = torch.tensor(self.labels.iloc[index], dtype=torch.float32)
            return data, label
    
    sd = SurvivalData("train.csv")
    model_input_columns = sd.data.columns.tolist()
    with open("input_columns.json", "w") as f:
        json.dump(model_input_columns, f)
    train_data, test_data = random_split(sd, [1 - TEST_SIZE, TEST_SIZE])
    train_dataloader = DataLoader(train_data, batch_size=50)
    test_dataloader = DataLoader(test_data, batch_size=50)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} device")
    
    input_size = sd.data.shape[1]
    model = nn.Sequential(
        nn.Linear(input_size, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 2)
    )

    model.to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 500
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for data, label in train_dataloader:
            data = data.to(device)
            label = label.to(device).long()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, label in test_dataloader:
            data = data.to(device)
            label = label.to(device).long()
            output = model(data)
            preds = torch.argmax(output, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    torch.save(model.state_dict(), "titanic_model.pt")


if __name__ == "__main__":
    main()

