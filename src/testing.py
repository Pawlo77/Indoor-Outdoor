"""Evaluate a PyTorch model on a test dataset."""

import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm

def evaluate_model(
    test_ds: Dataset, model: Module, batch_size: int = 32, device: str = "cuda"
) -> None:
    """
    Evaluate the given model on a test dataset.

    Parameters:
        test_ds: The dataset containing test images and labels.
        model: The PyTorch model to be evaluated.
        batch_size (optional): Number of samples per batch. Default is 32.
        device (optional): The device identifier to run the evaluation ('cuda' or 'cpu').
                           If 'cuda' is specified, uses cuda if available,
                           otherwise defaults to cpu.

    Returns:
        None. Prints the Accuracy, F1 Score, and ROC AUC score.
    """
    device = torch.device(
        device if torch.cuda.is_available() and device == "cuda" else "cpu"
    )
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    auc = roc_auc_score(all_labels, all_probs)

    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("ROC AUC:", auc)
