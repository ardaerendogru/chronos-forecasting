import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from chronos import ChronosPipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sktime.datasets import load_from_tsfile_to_dataframe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SIZES = ["tiny", "mini", "small", "base", "large"]

# Simple classifier model
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)
    
# Load the dataset
def load_dataset(dataset_name):
    allowed_datasets = ["ArrowHead", "BasicMotions", "GunPoint", "ItalyPowerDemand"]
    if dataset_name not in allowed_datasets:
        raise ValueError(f"dataset_name must be one of {allowed_datasets}")
    print(f"Loading {dataset_name} dataset...")
    from sktime.datasets import load_UCR_UEA_dataset

    X_train, y_train = load_UCR_UEA_dataset(name=dataset_name, split="train", return_X_y=True)
    X_test, y_test = load_UCR_UEA_dataset(name=dataset_name, split="test", return_X_y=True)
    print(f"Loaded {dataset_name} dataset!")

    # Convert to numpy arrays
    X_train = np.array([x.values for x in X_train.iloc[:, 0]])
    X_test = np.array([x.values for x in X_test.iloc[:, 0]])

    # Concatenate train and test sets, as we will split them later
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    return X, y

# Load the Chronos model
def load_chronos_model(model_size):
    if model_size not in MODEL_SIZES:
        raise ValueError(f"model_size must be one of {MODEL_SIZES}")
    print(f"Loading Chronos model ({model_size})...")
    model_name = f"amazon/chronos-t5-{model_size}"

    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map=DEVICE,
        torch_dtype=torch.bfloat16,
    )
    print(f"Loaded {model_name} model!")
    return pipeline

# Extract embeddings from a time series dataset using the Chronos model
def extract_chronos_embeddings(time_series_list, extractor):
    print(f"Extracting embeddings...")
    # Convert to torch tensor if needed
    if isinstance(time_series_list[0], np.ndarray):
        time_series_list = [torch.tensor(x) for x in time_series_list]
    
    embeddings, _ = extractor.embed(time_series_list)
    print(f"Extracted embeddings!")
    
    # Average pooling over the time dimension to get a fixed-size representation
    embeddings_pooled = embeddings.mean(dim=1)

    # Convert to float32 before moving to NumPy
    return embeddings_pooled.to(torch.float32).cpu().numpy()

# Train a simple classifier using the Chronos embeddings
def train_classifier(X_embeddings, y_labels, num_epochs, batch_size):
    label_encoder = LabelEncoder()
    y_labels_encoded = label_encoder.fit_transform(y_labels)
    X_embeddings = torch.FloatTensor(X_embeddings)
    y_labels = torch.LongTensor(y_labels_encoded)
    
    X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y_labels, test_size=0.3, random_state=42)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    num_classes = len(torch.unique(y_labels))
    model = SimpleClassifier(
        input_dim=X_embeddings.shape[1],
        num_classes=num_classes,
        dropout_rate=0.5
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Training loop
    best_test_acc = 0
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Training accuracy
            X_train_gpu = X_train.to(DEVICE)
            train_outputs = model(X_train_gpu)
            _, predicted = torch.max(train_outputs, 1)
            train_acc = (predicted.cpu() == y_train).float().mean().item()
            train_accuracies.append(train_acc)
            
            # Test accuracy
            X_test_gpu = X_test.to(DEVICE)
            test_outputs = model(X_test_gpu)
            _, predicted = torch.max(test_outputs, 1)
            test_acc = (predicted.cpu() == y_test).float().mean().item()
            test_accuracies.append(test_acc)
        
        # Update learning rate
        scheduler.step(train_loss)
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = model.state_dict()
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Training accuracy: {train_acc:.3f}')
            print(f'Test accuracy: {test_acc:.3f}')
            print(f'Loss: {train_loss/len(train_loader):.4f}\n')
    
    # Load best model
    model.load_state_dict(best_state)
    print(f'\nBest test accuracy: {best_test_acc:.3f}')

    # Plot accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy over Epochs')
    plt.legend()
    plt.show()
    
    return model

def classify_embeddings_from_chronos(dataset_name):
    X, y = load_dataset(dataset_name)

    model_size = "tiny"
    extractor = load_chronos_model(model_size)
    X_embeddings = extract_chronos_embeddings(X, extractor)

    num_epochs = 100
    batch_size = 16
    classifier = train_classifier(X_embeddings, y, num_epochs, batch_size)

if __name__ == "__main__":
    dataset_name = "ArrowHead" # "ArrowHead", "BasicMotions", "GunPoint", "ItalyPowerDemand"
    classify_embeddings_from_chronos(dataset_name)