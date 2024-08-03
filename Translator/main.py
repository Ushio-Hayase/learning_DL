import os
import time

import torch
import numpy as np
import torch.utils
import transformers
import tqdm
import torchmetrics
from data import Dataset
from model import Transformer

BATCH_SIZE = 200
EPOCHS = 20
MAX_LEN = 128
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = transformers.AutoTokenizer.from_pretrained("quantumaikr/KoreanLM")
EOS= tokenizer.eos_token_id
BOS = tokenizer.bos_token_id


def collate_fn(samples):
    x, y_1, y_2 = [], [], []
    for i in samples:
        x.append(tokenizer(i[0].item(), padding="max_length", add_special_tokens=False, truncation=True, max_length=MAX_LEN, return_tensors="pt")['input_ids'].squeeze())
        y_1.append(tokenizer(i[1].item() + tokenizer.eos_token, padding="max_length", add_special_tokens=False,truncation=True, max_length=MAX_LEN, return_tensors="pt")['input_ids'].squeeze())
        y_2.append(tokenizer(i[1].item(), padding="max_length", truncation=True,  max_length=MAX_LEN, return_tensors="pt")['input_ids'].squeeze())

    x = torch.stack(x)
    y_1 = torch.stack(y_1)
    y_2 = torch.stack(y_2)

    return x, y_1, y_2

if __name__ == "__main__":

    model = Transformer(256 , 512, tokenizer.vocab_size, MAX_LEN, device, tokenizer.pad_token_id).to(device=device)
    model.load_state_dict(torch.load("model.pt"))

    loss_object = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id).to(device=device)
    learning_rate = 1e-5
    

    optimizer = torch.optim.Adam(model.parameters(),learning_rate)
    acc_fn = torchmetrics.Accuracy(task="multiclass", num_classes=tokenizer.vocab_size, ignore_index=tokenizer.pad_token_id).to(device=device)
 

    dataset = np.load("data.npy", allow_pickle=True)
    dataset = Dataset(dataset, BATCH_SIZE)

    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    validation_size = int(dataset_size * 0.1)
    test_size = dataset_size - train_size - validation_size

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])


    train_dataloader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model.train()

    for epoch in range(EPOCHS):
        with tqdm.tqdm(train_dataloader, unit="batch") as tepoch:
            for (enc_in, tar_real, dec_in) in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")

                optimizer.zero_grad()

                enc_in, dec_in, tar_real = enc_in.to(device=device), dec_in.to(device=device), tar_real.to(device=device)


                out = model(enc_in, dec_in)
                loss = loss_object(out, tar_real)
            
                loss.backward()

                optimizer.step()
                
                
                acc = acc_fn(out, tar_real)

                tepoch.set_postfix(loss=loss.item(), acc=acc.item())

        with torch.no_grad():
            with tqdm.tqdm(valid_dataloader, unit="batch") as tepoch:
                for (enc_in, tar_real, dec_in) in tepoch:
                    tepoch.set_description(f"Valid - Epoch {epoch+1}")


                    optimizer.zero_grad()

                    enc_in, dec_in, tar_real = enc_in.to(device=device), dec_in.to(device=device), tar_real.to(device=device)


                    out = model(enc_in, dec_in)
                    loss = loss_object(out, tar_real)
                    
                    acc = acc_fn(out, tar_real)

                    tepoch.set_postfix(val_loss=loss.item(), val_acc=acc.item())

    with torch.no_grad():
        with tqdm.tqdm(test_dataloader, unit="batch") as tepoch:
            for (enc_in, tar_real, dec_in) in tepoch:
                tepoch.set_description(f"Test ")


                optimizer.zero_grad()

                enc_in, dec_in, tar_real = enc_in.to(device=device), dec_in.to(device=device), tar_real.to(device=device)


                out = model(enc_in, dec_in)
                loss = loss_object(out, tar_real)
                acc = acc_fn(out, tar_real)

                tepoch.set_postfix(test_loss=loss.item(), test_acc=acc.item())

            
    torch.save(model.state_dict(), "model.pt")

