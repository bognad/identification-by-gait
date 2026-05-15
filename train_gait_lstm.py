# train_gait_lstm.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from embeddings.gait_model import GaitLSTM
from dataset_gait import GaitDataset, collate_fn

def main():
    device = torch.device("cpu")  # у тебя cuda.is_available() = False

    dataset = GaitDataset("gait_sequences_41.npz")
    num_classes = len(dataset.label2idx)

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = GaitLSTM(input_dim=41, hidden_dim=128, emb_dim=128, num_layers=1, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 40
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0

        for x, y, lengths in loader:
            x = x.to(device)  # (B, T, 41)
            y = y.to(device)

            optimizer.zero_grad()
            emb, logits = model(x)            # logits: (B, num_classes)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total += x.size(0)

        print(f"Epoch {epoch}: loss={total_loss/total:.4f}, acc={total_correct/total:.4f}")

    torch.save(model.state_dict(), "gait_lstm.pth")
    print("Saved model to gait_lstm.pth")

if __name__ == "__main__":
    main()