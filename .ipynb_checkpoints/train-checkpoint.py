import torch
from torch.autograd import Variable
from tqdm import tqdm

def evaluate(model, data_loader,  criterion):
    correct = 0
    total = 0
    loss = 0
    for (image, target) in data_loader:
        total += len(target)
        output = model(image)
        loss += criterion(output, target)
        _, pred = torch.max(output, 1)
        correct += (pred == target).sum().item()

    return correct / total, loss


def train(train_data_loader, test_data_loader,
          model, optimizer, criterion,
          epochs=25):
    acc_train_hist =[]
    acc_test_hist = []
    loss_tr_hist = []
    loss_te_hist = []
    for epoch in range(epochs):
        with tqdm(total=60000, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for (image, target) in train_data_loader:
                model.train()
                optimizer.zero_grad()
                # with torch.no_grad():
                #     with torch.enable_grad():
                #         torch.set_grad_enabled(True)
                output = model(image)
                loss = criterion(output, target)
               # print("loss", loss)
                loss.backward()
                optimizer.step()
                pbar.update(target.shape[0])

            model.eval()
            with torch.no_grad():
                    acc_train, loss_tr= evaluate(model, train_data_loader, criterion)# evaluate model with training set
                    acc_test, loss_te = evaluate(model, test_data_loader, criterion) # evaluate model with test set
                    acc_test_hist.append(acc_test)
                    acc_train_hist.append(acc_train)
                    loss_tr_hist.append(loss_tr)
                    loss_te_hist.append(loss_te)
                    pbar.set_postfix(**{"train loss": loss_tr.item(), "test loss": loss_te.item(), "train acccuracy": acc_train,
                                "test accuracy": acc_test})
    return loss_tr_hist, loss_te_hist, acc_train_hist, acc_test_hist
