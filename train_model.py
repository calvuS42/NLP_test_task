import pandas as pd
import torch
from torch import optim, nn
from torch.utils.data import Dataset, random_split, DataLoader
import transformers
from transformers import BertTokenizerFast, BertForSequenceClassification
import timeit
from tqdm import tqdm 
import os


# define the device which is used for training 
device = "cuda" if torch.cuda.is_available() else "cpu"

# define hyperparameters
MAX_LENGTH = 256
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 2

# upload train and test data
data_folder = './data'
train_df = pd.read_csv(os.path.join(data_folder, "train.csv"))
test_df = pd.read_csv(os.path.join(data_folder, "test.csv"))

# define the model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)

# define the dataset for training loop
class ComplexityDataset(Dataset):
    def __init__(self, sentences, targets, tokenizer):
        self.encodings = tokenizer(sentences, padding=True, truncation=True, max_length=MAX_LENGTH)
        self.targets = targets
        
    def __getitem__(self, idx):
        out_dic = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        out_dic["targets"] = torch.tensor(self.targets[idx], dtype=torch.float)
        return out_dic
    
    def __len__(self):
        return len(self.targets)
    
# define dataset for submitting on test batch
class ComplexitySubmitDataset(Dataset):
    def __init__(self, sentences, tokenizer, ids):
        self.ids = ids
        self.encodings = tokenizer(sentences, padding=True, truncation=True, max_length=MAX_LENGTH)
        
    def __getitem__(self, idx):
        out_dic = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        out_dic["ids"] = self.ids[idx]
        return out_dic
    
    def __len__(self):
        return len(self.ids)

# define the RMSE loss 
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

# create train, val, test sets
# and dataloaders
dataset = ComplexityDataset(train_df["excerpt"].to_list(), train_df["target"].to_list(), tokenizer)
test_dataset = ComplexitySubmitDataset(test_df["excerpt"].to_list(), tokenizer, test_df["id"].to_list())

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [0.9, 0.1], generator=generator)

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

val_dataloader = DataLoader(dataset=val_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False)


# the training loop
# define the parameters and run the training process
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

start = timeit.default_timer() 
for epoch in tqdm(range(EPOCHS), position=0, leave=True):
    model.train()
    train_running_loss = 0 
    for idx, sample in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        input_ids = sample['input_ids'].to(device)
        attention_mask = sample['attention_mask'].to(device)
        targets = sample["targets"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_running_loss += loss.item()
    train_loss = train_running_loss / (idx + 1)

    model.eval()
    val_running_loss = 0 
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(val_dataloader, position=0, leave=True)):
            input_ids = sample['input_ids'].to(device)
            attention_mask = sample['attention_mask'].to(device)
            targets = sample["targets"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)

            val_running_loss += outputs.loss.item()
        val_loss = val_running_loss / (idx + 1)

    print("-"*30)
    print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
    print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
    print("-"*30)
torch.save(model.state_dict(), 'model.pt')
stop = timeit.default_timer()
print(f"Training Time: {stop-start:.2f}s")

# create the predictions for the test set
# in appropriate submission format
preds = []
ids = []
model.eval()
with torch.no_grad():
    for idx, sample in enumerate(tqdm(test_dataloader, position=0, leave=True)):
        input_ids = sample['input_ids'].to(device)
        attention_mask = sample['attention_mask'].to(device)
        ids.extend(sample["ids"])
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds.extend([float(i) for i in outputs["logits"].squeeze()])

submission_df = pd.DataFrame(list(zip(ids, preds)),
               columns =['id', 'target'])
submission_df.to_csv("submission.csv", index=False)
