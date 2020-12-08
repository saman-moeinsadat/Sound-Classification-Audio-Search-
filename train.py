import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch
import torch.nn as nn
import time
import copy
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import accuracy_score
import torchvision.models as models
from util import extract_code_label
from pathlib import Path


# The clip_gradient function does as its name sugest:
# clipping the gradient to prevent exploding gradiesnts
# effect.


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


# The bi-directional LSTM RNN network:

class BiLSTMRNN(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 in_dropout,
                 hidden_dropout):
        super(BiLSTMRNN, self).__init__()
        self.lstm = nn.LSTM(input_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            dropout=hidden_dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(in_dropout)

    def forward(self, input_data):

        dropped = self.dropout(input_data)
        outputs, (hidden, cell) = self.lstm(dropped)
        predictions = self.fc(self.dropout(outputs[-1]))
        return predictions


# train_val function trains and then validates the network:

def train_val(
    criterion,
    data='data.npy', epochs_number=40,
    hidden_dim=512, n_layers=2, in_dropout=0.2, hidden_dropout=0.2
):
    since = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Converts the numpy arrays to torch arrays.
    data = torch.from_numpy(np.load(data))

    # Prepares the train/val dataset and data loaders.
    label, cls = extract_code_label()
    (
        train_dataset, val_dataset, train_data_loader, val_data_loader
    ) = prepare_dataset_dataloader(data, torch.from_numpy(label))

    data_loaders = dict(
        zip(['train', 'val'], [train_data_loader, val_data_loader])
    )
    datasets = dict(zip(['train', 'val'], [train_dataset, val_dataset]))

    # preparing the model:
    model = model_prepare(len(cls))
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Reducing learning-rate after a specific epoch:
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[round(epochs_number*x) for x in [0.65, 1]],
        gamma=0.1
    )
    scheduler.last_epoch = epochs_number

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs_number):
        t1 = time.time()
        print('Epoc {}/{}'.format(epoch, epochs_number - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            flag = False
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in data_loaders[phase]:

                # Using torch's Variable for closing the autograd
                # route.
                labels = Variable(labels)
                inputs = Variable(inputs.float(), requires_grad=True)
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    # Here the detections are stacked for
                    # measuring the metrics
                    if not flag:
                        detections = preds
                        labels_all = labels
                        flag = True
                    else:

                        detections = torch.cat((detections, preds), 0)
                        labels_all = torch.cat((labels_all, labels), 0)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        clip_gradient(model, 0.5)
                        optimizer.step()
                running_loss += loss.item() * inputs.size()[0]
                running_corrects += torch.sum(preds == labels).item()
            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_corrects / len(datasets[phase])
            epoch_f1 = f1_score(labels_all, detections, average='weighted')
            epoch_recall = recall_score(
                labels_all, detections, average='weighted'
            )
            epoch_precision = precision_score(
                labels_all, detections, average='weighted'
            )
            epoch_acc_sk = accuracy_score(
                labels_all, detections, normalize=True
            )
            print("""
                {}  ==>  Loss: {:.4f}   Accuracy: {:.2f} %   Recall: {:.2f} %
                Precision: {:.2f} %   F1_score: {:.2f} %  Accuracy_sk: {:.2f} %
            """.format(
                phase.title(), epoch_loss, epoch_acc * 100, epoch_recall * 100,
                epoch_precision * 100, epoch_f1 * 100, epoch_acc_sk * 100
            ))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        scheduler.step()
        t2 = time.time()
        print('Epoch running time in {:.0f}m {:.0f}s'.format(
            (t2 - t1) // 60, (t2 - t1) % 60))
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Accuracy: {:.2f} %'.format(best_acc * 100))
    model.load_state_dict(best_model_wts)

    return model


# prepare_dataset_dataloader takes in numpy arrays
# as input and returns train/validate datasets and dataloaders.


def prepare_dataset_dataloader(
    data_tensor, label_tensor, bs=16, val_train_ratio=0.7
):
    dataset = TensorDataset(data_tensor, label_tensor)
    train_length = int(val_train_ratio * len(data_tensor))
    val_length = len(data_tensor) - train_length
    train_dataset, val_dataset = random_split(
        dataset, [train_length, val_length]
    )
    train_data_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=bs, shuffle=True)
    return train_dataset, val_dataset, train_data_loader, val_data_loader


# Prepare the resnet18 model and costumizes the input/output.
def model_prepare(number_of_classes=12):
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model.fc = nn.Linear(
        in_features=512, out_features=number_of_classes, bias=True
    )
    return model


# Concept from : https://github.com/peimengsui/semi_supervised_mnist
# After training the model with augmented data the 95.95 Acc. on the
# validation set was achieved. For the case of abundancy in un-lebelled
# data semi-supervised learning has been shown to be very effective.
# In this case the 99.25 Acc. was achieved after semi-supervised
# learning.

def train_val_semi_supervied(
    criterion,
    data='data.npy', epochs_number=150,
    data_unlabelled='data_unlabelled.npy'

):
    PATH = str((Path(__file__).parent).resolve())
    since = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.from_numpy(np.load(data)).type(torch.FloatTensor)
    label, cls = extract_code_label()
    (
        train_dataset, val_dataset, train_data_loader, val_data_loader
    ) = prepare_dataset_dataloader(data, torch.from_numpy(label))
    data_unlabelled = torch.from_numpy(np.load(data_unlabelled)).float()
    train_unlabelled_dst = TensorDataset(data_unlabelled)
    unlabelled_dataloader = DataLoader(
        train_unlabelled_dst, batch_size=64, shuffle=True
    )
    model = model_prepare(len(cls))
    model.load_state_dict(torch.load(PATH+'/model_weights.pt'))
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Instead of using current epoch we use a "step" variable to calculate
    # alpha_weight. This helps the model converge faster.
    step = 100
    for epoch in range(epochs_number):
        t1 = time.time()
        print('Epoc {}/{}'.format(epoch, epochs_number - 1))
        print('-' * 10)
        for batch_idx, input_unlabelled in enumerate(unlabelled_dataloader):

            # Forward Pass to get the pseudo labels.
            input_unlabelled = Variable(
                input_unlabelled[0], requires_grad=True
            )
            input_unlabelled = input_unlabelled.to(device)
            model.eval()
            output_unlabeled = model(input_unlabelled)
            _, pseudo_labeled = torch.max(output_unlabeled, 1)

            # Now calculate the unlabeled loss using the pseudo label
            model.train()
            output_pseudo = model(input_unlabelled)
            unlabeled_loss = alpha_weight(step) * criterion(
                output_pseudo, pseudo_labeled
            )
            optimizer.zero_grad()
            unlabeled_loss.backward()
            clip_gradient(model, 0.5)
            optimizer.step()

            # For every 50 batches train one epoch on labeled data
            if batch_idx % 25 == 0:
                for inputs, labels in train_data_loader:
                    model.train()

                    # Using torch's Variable for closing the autograd
                    # route.
                    labels = Variable(labels)
                    inputs = Variable(inputs.float(), requires_grad=True)
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(True):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)

                        loss = criterion(outputs, labels)
                        loss.backward()
                        clip_gradient(model, 0.5)
                        optimizer.step()
                # Now we increment step by 1
                step += 1
        running_loss = 0.0
        model.eval()
        flag = False
        with torch.no_grad():
            for inputs, labels in val_data_loader:
                labels = Variable(labels)
                inputs = Variable(inputs.float(), requires_grad=False)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                # Here the detections are stacked for
                # measuring the metrics
                if not flag:

                    detections = preds
                    labels_all = labels
                    flag = True
                else:

                    detections = torch.cat((detections, preds), 0)
                    labels_all = torch.cat((labels_all, labels), 0)

                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size()[0]
        epoch_loss = running_loss / len(val_dataset)
        epoch_f1 = f1_score(labels_all, detections, average='weighted')
        epoch_recall = recall_score(
            labels_all, detections, average='weighted'
        )
        epoch_precision = precision_score(
            labels_all, detections, average='weighted'
        )
        epoch_acc = accuracy_score(
            labels_all, detections, normalize=True)

        print("""
            Validation  ==>  Loss: {:.6f}   Recall: {:.2f} %
            Precision: {:.2f} %   F1_score: {:.2f} %  Accuracy: {:.2f} %
        """.format(
            epoch_loss, epoch_recall * 100,
            epoch_precision * 100, epoch_f1 * 100, epoch_acc * 100
        ))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        model.train()
        t2 = time.time()
        print('Epoch running time:  {:.0f}m {:.0f}s'.format(
            (t2 - t1) // 60, (t2 - t1) % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Accuracy: {:.2f} %'.format(best_acc * 100))
    model.load_state_dict(best_model_wts)

    return model


def alpha_weight(step, T1=100, T2=500, af=3):

    if step < T1:
        return 0.0
    elif step > T2:
        return af
    else:
        return ((step-T1) / (T2-T1))*af


if __name__ == "__main__":
    PATH = str((Path(__file__).parent).resolve())
    criterion = nn.CrossEntropyLoss()
    best_model = train_val(criterion)
    torch.save(best_model.state_dict(), PATH+'/model_weights.pt')
    train_val_semi_supervied(criterion)
    best_model_semi = train_val_semi_supervied(criterion)
    torch.save(best_model_semi.state_dict(), PATH+'/model_weights_semi.pt')
