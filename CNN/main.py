import torch, tqdm
import torch.utils.data
from data import train_dataset, test_dataset
from model import CNN

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
EPOCHS = 2
BATCH_SIZE = 32
learning_rate = 0.01

def main():
    train_dataloader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset)

    model = CNN()
    loss_func = torch.nn.CrossEntropyLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(EPOCHS):
        with tqdm.tqdm(train_dataloader, unit="batch", ncols=200 ,mininterval=1) as tepoch:
            for x, y in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")

                optimizer.zero_grad()
                output = model(x)
                loss = loss_func(output, y)
                loss.backward()
                predict = torch.argmax(output, -1) == y
                acc = predict.float().mean()
                optimizer.step()

                tepoch.set_postfix(loss= loss.item(), acc= acc.item())

    with torch.no_grad():
        for x, y in test_dataloader:
            prediction = model(x)
            correct_prediction = torch.argmax(prediction, 1) == y
            accuracy = correct_prediction.float().mean()
            print('Accuracy:', accuracy.item())

    torch.save(model.state_dict(), "model.pt")

if __name__ == "__main__":
    main()
                