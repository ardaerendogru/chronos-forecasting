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
from sktime.datasets import load_UCR_UEA_dataset
from catboost import CatBoostClassifier

class Classifier(nn.Module):
    def __init__(self, chronos_model, dataset_name, dropout, finetune, batch_size, device):
        super().__init__()
        self.device = device  
        self.pipeline = self._load_chronos_model(chronos_model)
        
        input_dim = {
            "tiny": 256,
            "mini": 384,
            "small": 512,
            "base": 768,
            "large": 1024,
        }.get(chronos_model)
        self.label_encoder = LabelEncoder()

        x_train, y_train, x_test, y_test = self._load_dataset(dataset_name)
        x_train = np.array([x.values for x in x_train.iloc[:, 0]])
        x_test = np.array([x.values for x in x_test.iloc[:, 0]])

        self.y_train = self.label_encoder.fit_transform(y_train)
        self.y_test = self.label_encoder.transform(y_test)
        
        y_train = torch.tensor(self.y_train,
                               dtype=torch.long).to(self.device)
        y_test = torch.tensor(self.y_test,
                              dtype=torch.long).to(self.device)
        
        
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)



        context_tensor = self.pipeline._prepare_and_validate_context(context=x_train_tensor)
        x_train_tensor, x_train_attention_mask, tokenizer_state = (
            self.pipeline.tokenizer.context_input_transform(context_tensor)
        )

        context_tensor = self.pipeline._prepare_and_validate_context(context=x_test_tensor)
        x_test_tensor, x_test_attention_mask, tokenizer_state = (
            self.pipeline.tokenizer.context_input_transform(context_tensor)
        )
        self.x_train_tensor = x_train_tensor
        self.x_test_tensor = x_test_tensor
        self.x_test_attention_mask = x_test_attention_mask
        self.x_train_attention_mask = x_train_attention_mask
        self.num_classes = len(torch.unique(y_train))
        
        self.classification_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, self.num_classes)
        ).to(self.device)
        
        self.finetune = finetune
        
        if finetune:
            self.optimizer = optim.AdamW(
                list(self.classification_head.parameters()) + list(self.pipeline.model.model.encoder.parameters()),
                lr=5e-5,
            )
        else:
            self.optimizer = optim.AdamW(
                list(self.classification_head.parameters()),
                lr=5e-3,
            )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        self.criterion = nn.CrossEntropyLoss()

        self.train_dataset = TensorDataset(x_train_tensor, x_train_attention_mask, y_train)
        self.test_dataset  = TensorDataset(x_test_tensor, x_test_attention_mask, y_test)
        self.train_loader  = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.test_loader   = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def _load_dataset(self, dataset_name):
        print(f"Loading {dataset_name} dataset...")
        X_train, y_train = load_UCR_UEA_dataset(name=dataset_name, split="train", return_X_y=True)
        X_test, y_test = load_UCR_UEA_dataset(name=dataset_name, split="test", return_X_y=True)
        return X_train, y_train, X_test, y_test

    def _load_chronos_model(self, model_size):
        print(f"Loading Chronos model ({model_size})...")
        model_name = f"amazon/chronos-t5-{model_size}"
        pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=self.device,
            torch_dtype=torch.float32,
        )
        print(f"Loaded {model_name} model!")
        return pipeline
    def _train(self, num_epochs=50):
        """
        Train the classification head (and encoder if finetuning) for num_epochs.
        """
        for epoch in range(num_epochs):
            self.train()  
            running_loss = 0.0

            
            for inputs, attention_mask, labels in self.train_loader:
                inputs = inputs.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                
                
                
                self.optimizer.zero_grad()
                
                features = self.pipeline.model.encode(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                ).mean(dim=1)
                outputs = self.classification_head(features)
                
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            
            avg_train_loss = running_loss / len(self.train_loader.dataset)
            val_loss, val_accuracy = self.evaluate()
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.4f}")
            self.scheduler.step(val_loss)
    def train_catboost(self):
        
        train_feature_dataset = TensorDataset(self.x_train_tensor, self.x_train_attention_mask)
        train_feature_loader = DataLoader(train_feature_dataset, batch_size=32, shuffle=False) 

        test_feature_dataset = TensorDataset(self.x_test_tensor, self.x_test_attention_mask)
        test_feature_loader = DataLoader(test_feature_dataset, batch_size=32, shuffle=False)   


        train_features_list = []
        test_features_list = []

        self.pipeline.model.to(self.device) 

        self.pipeline.model.eval() 
        with torch.no_grad(): 
            print("Extracting training features...")
            for batch_idx, (inputs, attention_mask) in enumerate(train_feature_loader):
                inputs = inputs.to(self.device)
                attention_mask = attention_mask.to(self.device)

                batch_features = self.pipeline.model.encode(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                ).mean(dim=1) 

                train_features_list.append(batch_features.cpu().numpy()) 

                if batch_idx % 10 == 0: 
                    print(f"  Processed training batch {batch_idx}")

            print("Extracting testing features...")
            for batch_idx, (inputs, attention_mask) in enumerate(test_feature_loader):
                inputs = inputs.to(self.device)
                attention_mask = attention_mask.to(self.device)

                batch_features = self.pipeline.model.encode(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                ).mean(dim=1) 

                test_features_list.append(batch_features.cpu().numpy()) 
                if batch_idx % 10 == 0: 
                    print(f"  Processed testing batch {batch_idx}")


        
        features_train_np = np.concatenate(train_features_list, axis=0)
        features_test_np  = np.concatenate(test_features_list, axis=0)
        y_train_np = self.y_train 
        y_test_np  = self.y_test


        print("Training CatBoost...")
        self.catboost_model = CatBoostClassifier(
                            iterations=1000,  
                            learning_rate=0.03, 
                            depth=6, 
                            l2_leaf_reg=3, 
                            loss_function='MultiClass',
                            eval_metric='Accuracy',
                            random_seed=42,
                            verbose=True,
                            task_type="GPU",
                            devices='0',
                        )

        self.catboost_model.fit(features_train_np, y_train_np,
                                 eval_set=(features_test_np, y_test_np),
                                 early_stopping_rounds=15, 
                                 verbose=True) 
    def evaluate(self):
        """
        Evaluate the model on the test dataset.
        Returns average loss and accuracy.
        """
        self.eval()
        total_loss = 0.0
        correct_predictions = 0
        
        with torch.no_grad():
            for inputs, attention_mask, labels in self.test_loader:
                inputs = inputs.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                features = self.pipeline.model.encode(
                        input_ids=inputs.to(self.device),
                        attention_mask=attention_mask.to(self.device),
                    ).mean(dim=1)
                
                outputs = self.classification_head(features)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                
                preds = torch.argmax(outputs, dim=1)
                correct_predictions += (preds == labels).sum().item()
                
        avg_loss = total_loss / len(self.test_loader.dataset)
        accuracy = correct_predictions / len(self.test_loader.dataset)
        return avg_loss, accuracy

import contextlib  

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    datasets = ["ECG5000", "UWaveGestureLibraryX", "FordA"]
    model_sizes = ["tiny", "mini", "small"]
    finetune_options = [True, False]
    
    
    dropout = 0.5
    batch_size = 4
    num_epochs = 100

    results_filename = "results.txt"
    
    
    with open(results_filename, "w", buffering=1) as f:
        f.write("Chronos Classification Experiments Results\n")
        f.write("============================================\n")
    
    experiment_counter = 1
    for dataset in datasets:
        for model_size in model_sizes:
            for finetune in finetune_options:
                experiment_info = f"Experiment {experiment_counter}: Dataset={dataset}, Model={model_size}, Finetune={finetune}"
                print(f"Starting {experiment_info}")
                
                with open(results_filename, "a", buffering=1) as f:
                    f.write("\n" + "=" * 50 + "\n")
                    f.write(experiment_info + "\n")
                
                
                with open(results_filename, "a", buffering=1) as f, \
                     contextlib.redirect_stdout(f), \
                     contextlib.redirect_stderr(f):
                    
                    clf = Classifier(model_size, dataset, dropout, finetune, batch_size, device)
                    
                    clf._train(num_epochs=num_epochs)
                    
                    test_loss, test_accuracy = clf.evaluate()
                    print(f"Final Evaluation - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
                
                print(f"Finished {experiment_info}")
                experiment_counter += 1

if __name__ == "__main__":
    main()