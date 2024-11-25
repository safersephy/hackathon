import os
import torch
import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_models(folder_path):
    """
    Load all PyTorch models from the specified folder.
    """
    models = []
    for file in os.listdir(folder_path):
        if file.endswith(".pt"):  # Ensure models are .pt files
            model_path = os.path.join(folder_path, file)
            model = torch.load(model_path)
            model.eval()  # Set to evaluation mode
            models.append(model)
    return models

def majority_vote(predictions):
    """
    Perform majority voting for a list of predictions.
    """
    vote_counts = Counter(predictions)
    return vote_counts.most_common(1)[0][0]

def predict_with_voting(models, dataloader, device):
    """
    Run the dataset through each model, collect predictions, and vote on the final label.
    """
    all_predictions = []
    for model in models:
        model_predictions = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)  # Get predicted class
                model_predictions.extend(predicted.cpu().numpy())
        all_predictions.append(model_predictions)
    
    # Transpose predictions to group predictions for each sample
    all_predictions = np.array(all_predictions).T
    
    # Apply majority voting
    final_predictions = [majority_vote(sample_predictions) for sample_predictions in all_predictions]
    return final_predictions

def calculate_class_accuracy(true_labels, predictions, num_classes):
    """
    Calculate the percentage of correct predictions for each class.
    """
    class_accuracies = {}
    for class_label in range(num_classes):
        indices = (true_labels == class_label)
        correct = (predictions[indices] == true_labels[indices]).sum()
        total = indices.sum()
        accuracy = correct / total if total > 0 else 0
        class_accuracies[f"Class {class_label}"] = accuracy * 100  # Convert to percentage
    return class_accuracies

# Main Execution
if __name__ == "__main__":
    # Configuration
    models_folder = "path/to/models/folder"
    dataset_path = "path/to/dataset.csv"
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 5  # Number of classes in your dataset
    
    # Load dataset
    dataset = pd.read_csv(dataset_path)  # Adjust if your dataset format differs
    
    # Extract features and labels
    if 'label' not in dataset.columns:
        raise ValueError("Dataset must contain a 'label' column for comparison.")
    
    features = dataset.drop(columns=['label']).values
    labels = dataset['label'].values
    
    # Convert dataset to PyTorch TensorDataset
    data_tensor = torch.tensor(features, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(data_tensor, label_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Load models
    models = load_models(models_folder)
    if not models:
        print("No models found in the specified folder.")
        exit(1)
    
    # Move models to device
    for model in models:
        model.to(device)
    
    # Predict with majority voting
    final_predictions = predict_with_voting(models, dataloader, device)
    
    # Evaluate results
    accuracy = accuracy_score(labels, final_predictions)
    conf_matrix = confusion_matrix(labels, final_predictions)
    class_accuracies = calculate_class_accuracy(labels, final_predictions, num_classes)
    
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nPer-Class Accuracy:")
    for class_label, class_accuracy in class_accuracies.items():
        print(f"{class_label}: {class_accuracy:.2f}%")
    
    # Save final predictions and metrics
    output_df = pd.DataFrame({
        "Sample_Index": range(len(final_predictions)),
        "True_Label": labels,
        "Predicted_Label": final_predictions
    })
    output_path = "final_predictions_with_metrics.csv"
    output_df.to_csv(output_path, index=False)
    
    # Save class accuracies to a separate file
    class_accuracy_df = pd.DataFrame(list(class_accuracies.items()), columns=["Class", "Accuracy (%)"])
    class_accuracy_path = "class_accuracies.csv"
    class_accuracy_df.to_csv(class_accuracy_path, index=False)
    
    print(f"Final predictions saved to {output_path}.")
    print(f"Class accuracies saved to {class_accuracy_path}.")
