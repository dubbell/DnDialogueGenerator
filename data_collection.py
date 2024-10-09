from datasets import load_dataset
from torch.utils.data import Dataset


class DialogueDataset(Dataset):
    def __init__(self, datasets, dataset_labels):
        self.datasets = datasets
        self.dataset_labels = dataset_labels

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)
    
    def __getitem__(self, idx):
        count = 0
        for dataset, dataset_label in zip(self.datasets, self.dataset_labels):
            if count + len(dataset) >= idx:
                count += len(dataset)
                continue
            
            item = dataset[idx - count].copy()
            item["type"] = dataset_label
            return item
    

def train_val_test(dataset_names, train_size = 0.8, val_size = 0.1, test_size = 0.1):
    assert train_size + val_size + test_size == 1, "train_size, val_size and test_size must add up to 1"
    train_sets, val_sets, test_sets = [], [], []
    dataset_labels = []

    if "romantic" in dataset_names:
        train_valtest = load_dataset("AlekseyKorshuk/synthetic-romantic-characters")["train"].train_test_split(test_size=val_size + test_size)
        val_test = train_valtest["test"].train_test_split(test_size = test_size / (test_size + val_size))
        train_sets.append(train_valtest["train"])
        val_sets.append(val_test["train"])
        test_sets.append(val_test["test"])
        dataset_labels.append("romantic")
    
    if "friendly" in dataset_names:
        train_valtest = load_dataset("AlekseyKorshuk/synthetic-friendly-characters")["train"].train_test_split(test_size=val_size + test_size)
        val_test = train_valtest["test"].train_test_split(test_size = test_size / (test_size + val_size))
        train_sets.append(train_valtest["train"])
        val_sets.append(val_test["train"])
        test_sets.append(val_test["test"])
        dataset_labels.append("romantic")
    
    if "fight" in dataset_names:
        train_valtest = load_dataset("AlekseyKorshuk/synthetic-fight-characters")["train"].train_test_split(test_size=val_size + test_size)
        val_test = train_valtest["test"].train_test_split(test_size = test_size / (test_size + val_size))
        train_sets.append(train_valtest["train"])
        val_sets.append(val_test["train"])
        test_sets.append(val_test["test"])
        dataset_labels.append("romantic")
    
    return DialogueDataset(train_sets, dataset_labels), DialogueDataset(val_sets, dataset_labels), DialogueDataset(test_sets, dataset_labels)
