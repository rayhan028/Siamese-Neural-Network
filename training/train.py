import torch

def train(model, loader, criterion, optimizer, epochs=5, mode="pair"):
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            if mode == "pair":
                img1, img2, label = [x.cuda() for x in batch]
                out1, out2 = model(img1, img2)
                loss = criterion(out1, out2, label)
            else:  # triplet
                img1, img_pos, img_neg = [x.cuda() for x in batch]
                out1, out_pos, out_neg = model(img1, img_pos, img_neg)
                loss = criterion(out1, out_pos, out_neg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {losses[-1]:.4f}")
    return losses
