import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

class WordDifficultyDataset(Dataset):
    def __init__(self, words, labels, tokenizer, max_length=10):
        self.words = words
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(word,
                                  truncation=True,
                                  padding='max_length',
                                  max_length=self.max_length,
                                  return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def encode_labels(labels):
    label_map = {"Easy": 0, "Medium": 1, "Hard": 2}
    return [label_map[label] for label in labels]

def main():
    # Sample dataset
    data = {
        'word': ['simple', 'education', 'photosynthesis', 'and', 'university'],
        'difficulty': ['Easy', 'Medium', 'Hard', 'Easy', 'Medium']
    }
    df = pd.DataFrame(data)
    labels = encode_labels(df['difficulty'].tolist())

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = WordDifficultyDataset(df['word'].tolist(), labels, tokenizer)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=2,
        logging_dir='./logs',
        logging_steps=5,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained("bert_difficulty_model")
    tokenizer.save_pretrained("bert_difficulty_model")

if __name__ == "__main__":
    main()
