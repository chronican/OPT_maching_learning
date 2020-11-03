import torch
from tqdm import tqdm
import time
def evaluate(model, data_loader,  criterion):
    """
    Evaluate given network model with given data set and parameters
    :param model: network model to be evaluated
    :param data_loader: data loader that contains image, target for evaluating
    :param criterion: loss function
    :return correct/total: task accuracy
    :return loss: task loss
    """
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
          epochs=25 , ifsparsity=False):
    """
    Train network model with given parameters
    :param train_data_loader: data loader for training set
    :param test_data_loader: data load for test set
    :param model: network model to be trained
    :param optimizer: optimizer for training
    :param criterion: loss function for optimizer
    :param epochs: number of training epochs
    :param ifsparsity: boolean flag to calculate sparsity or not
    :return loss_tr_hist: list of training loss in each epoch
    :return loss_te_hist: list of test(validation) loss in each epoch
    :return acc_train_hist: list of accuracies of training set in each epoch
    :return acc_test_hist: list of accuracies of testing set in each epoch
    :return sparsity: the sparsity of the parameters of network
    """
    acc_train_hist =[]
    acc_test_hist = []
    loss_tr_hist = []
    loss_te_hist = []
    for epoch in range(epochs):
        with tqdm(total=60000, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for (image, target) in train_data_loader:
                model.train()
                optimizer.zero_grad()
                output = model(image)
                loss = criterion(output, target)
                loss.backward()
                sparsity=optimizer.step(ifsparsity)
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
    if ifsparsity:
        return loss_tr_hist, loss_te_hist, acc_train_hist, acc_test_hist, sparsity
    return loss_tr_hist, loss_te_hist, acc_train_hist, acc_test_hist


def train_time(train_data_loader,
          model, optimizer, criterion,
          epochs=25):
    """
    Train network model with given parameters and record time duration
    :param train_data_loader: data loader for training set
    :param model: network model to be trained
    :param optimizer: optimizer for training
    :param criterion: loss function for optimizer
    :param epochs: number of training epochs
    :print time duration of the training process
    """
    start_time = time.time()
    for epoch in range(epochs):
            for (image, target) in train_data_loader:
                model.train()
                optimizer.zero_grad()
                output = model(image)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
    print("--- %s seconds ---" % (time.time() - start_time))
