from datasets import load_dataset
from torch.utils.data import Dataset
import torch


class DialogueDataset(Dataset):
    def __init__(self, datasets, dataset_labels, tokenizer):
        self.data = []
        for dataset, dataset_label in zip(datasets, dataset_labels):
            for item in dataset:                
                categories = ','.join(item["categories"]) + "," + dataset_label
                personalities = ','.join(item["personalities"])
                conversations = '\n'.join([line["content"] for line in item["conversation"]])

                tokens = tokenizer.encode(categories + "<|endoftext|>" + personalities + "<|endoftext|>" + conversations)

                self.data.append(tokens)
        
        longest = max([len(seq) for seq in self.data])
        for index in range(len(self.data)):
            self.data[index].extend([tokenizer.pad_token] * (longest - len(self.data[index])))
            self.data[index] = torch.tensor(self.data[index])


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    

def training_set(dataset_names, tokenizer):
    train_sets = []
    dataset_labels = []

    if "romantic" in dataset_names:
        train_sets.append(load_dataset("AlekseyKorshuk/synthetic-romantic-characters")["train"])
        dataset_labels.append("romantic")

    if "friendly" in dataset_names:
        train_sets.append(load_dataset("AlekseyKorshuk/synthetic-friendly-characters")["train"])
        dataset_labels.append("friendly")

    if "fight" in dataset_names:
        train_sets.append(load_dataset("AlekseyKorshuk/synthetic-fight-characters")["train"])
        dataset_labels.append("fight")
    
    return DialogueDataset(train_sets, dataset_labels, tokenizer)
