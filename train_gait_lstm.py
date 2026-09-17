# train_gait_lstm.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from embeddings.gait_model import GaitLSTM
from dataset_gait import GaitDataset, collate_fn

def main():
    device = torch.device("cpu")

    dataset = GaitDataset("gait_sequences_41_BULG_2.npz")
    num_classes = len(dataset.label2idx)
    print("num_classes:", num_classes)

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = GaitLSTM(
        input_dim=41,
        hidden_dim=128,
        emb_dim=128,
        num_layers=2,
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20

    # Early stopping параметры
    best_acc = 0.1
    patience = 3          # сколько эпох ждать улучшения acc перед остановкой
    epochs_no_improve = 0

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

        train_loss = total_loss / total
        train_acc = total_correct / total

        print(f"Epoch {epoch}: loss={train_loss:.4f}, acc={train_acc:.4f}")

        # Early stopping logic
        if train_acc > best_acc:
            best_acc = train_acc
            epochs_no_improve = 0
            # можно сохранять лучшую модель
            torch.save(model.state_dict(), "gait_lstm_best.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Best acc: {best_acc:.4f}")
            break

    # Сохраняем последнюю версию модели
    torch.save(model.state_dict(), "gait_lstm.pth")
    print("Saved model to gait_lstm.pth")

if __name__ == "__main__":
    main()