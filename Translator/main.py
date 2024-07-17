import os
import time

import torch
import numpy as np
import torch.utils
import transformers
import tqdm
from data import Dataset
from model import Transformer

BATCH_SIZE = 64
EPOCHS = 5
MAX_LEN = 64
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = transformers.AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")

def collate_fn(samples):

    for idx,i in enumerate(samples):
        if idx == 0:
            x = tokenizer(i[0].item(), padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")['input_ids']
            y = tokenizer(i[1].item(), padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")['input_ids']
        else:
            x = torch.concat([x,tokenizer(i[0].item(), padding="max_length", truncation=True, max_length=64, return_tensors="pt")['input_ids']], dim=0)
            y = torch.concat([y,tokenizer(i[1].item(), padding="max_length", truncation=True, max_length=64, return_tensors="pt")['input_ids']], dim=0)

    return x, y

def acc_fn(y_pred, y_true):
    y_pred = torch.argmax(y_pred, dim=1)
    acc = torch.eq(y_pred, y_true).float().mean().item()
    return acc

if __name__ == "__main__":

    model = Transformer(256 , 512, tokenizer.vocab_size, MAX_LEN, device).to(device=device)

    loss_object = torch.nn.CrossEntropyLoss().to(device=device)
    learning_rate = 0.01

    optimizer = torch.optim.Adam(model.parameters(),learning_rate, betas=(0.9, 0.98),
                                        eps=1e-9)


    dataset = np.load("Translator/data/data.npy", allow_pickle=True)
    dataset = Dataset(dataset, BATCH_SIZE)

    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    validation_size = int(dataset_size * 0.1)
    test_size = dataset_size - train_size - validation_size

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])


    train_dataloader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    for epoch in range(EPOCHS):
        with tqdm.tqdm(train_dataloader, unit="batch") as tepoch:
            for (enc_in, tar) in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")



                optimizer.zero_grad()

                dec_in = tar[:, :-1]
                tar_real = tar[:, 1:]

                enc_in, dec_in, tar_real = enc_in.to(device=device), dec_in.to(device=device), tar_real.to(device=device)

                out = model(enc_in, dec_in)
                loss = loss_object(out, tar_real)
                loss.backward()
                optimizer.step()
                
                acc = acc_fn(out, tar_real)


                tepoch.set_postfix(loss=loss.item(), acc=acc)

        with torch.no_grad():
            with tqdm.tqdm(valid_dataloader, unit="batch") as tepoch:
                for (enc_in, tar) in tepoch:
                    tepoch.set_description(f"Valid : Epoch {epoch+1}")

                    enc_in, tar = enc_in.cuda(), tar.cuda()

                    optimizer.zero_grad()

                    dec_in = tar[:, :-1]
                    tar_real = tar[:, 1:]

                    out = model(enc_in, dec_in)
                    loss = loss_object(out, tar_real)
                    
                    acc = acc_fn(out, tar_real)

                    tepoch.set_postfix(val_loss=loss.item(), val_acc=acc)

    with torch.no_grad():
        with tqdm.tqdm(test_dataloader, unit="batch") as tepoch:
            for (enc_in, tar) in tepoch:
                tepoch.set_description(f"Test : ")

                enc_in, tar = enc_in.cuda(), tar.cuda()

                optimizer.zero_grad()

                dec_in = tar[:, :-1]
                tar_real = tar[:, 1:]

                out = model(enc_in, dec_in)
                loss = loss_object(out, tar_real)
                
                acc = acc_fn(out, tar_real)

                tepoch.set_postfix(test_loss=loss.item(), test_acc=acc)

            

