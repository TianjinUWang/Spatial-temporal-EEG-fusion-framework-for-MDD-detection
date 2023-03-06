import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, sensitivity_score, specificity_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

from model import Transformer
from data_loader import data_generator
from args import Config, Path


def specificity(y_true, y_pred, n=5):
    spec = []
    con_mat = confusion_matrix(y_true, y_pred)  # Each row is the ground truth, and each column is the precision
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spec1 = tn / (tn + fp)
        spec.append(spec1)
    average_specificity = np.mean(spec)
    return average_specificity


def test(model, test_loader, config):
    model.eval()

    pred = []
    label = []

    with torch.no_grad():
        loop = tqdm(enumerate(test_loader), total=len(test_loader))
        for batch_idx, (data, target) in loop:
            data = data.to(config.device)
            target = target.to(config.device)
            data, target = Variable(data), Variable(target)

            output = model(data)

            pred.extend(np.argmax(output.data.cpu().numpy(), axis=1))
            label.extend(target.data.cpu().numpy())

        accuracy = accuracy_score(label, pred, normalize=True, sample_weight=None)
        average_sensitivity = con_mat[i, i] / np.sum(con_mat[i, :])
        average_specificity = specificity(label, pred, n=5)

        print('ACC: %.4f' % accuracy,'Sens: %.4f' % average_sensitivity, 'Spec: %.4f' % average_specificity)

        con_mat = confusion_matrix(label, pred)

    return accuracy, average_sensitivity, average_specificity, con_mat


def evaluate(config, path):
    dataset, labels, val_loader = data_generator(path_labels=path.path_labels, path_dataset=path.path_TF)

    kf = StratifiedKFold(n_splits=config.num_fold, shuffle=True, random_state=0)

    ACC = 0
    Sens = 0
    Spec = 0
    Confusion_mat = np.zeros([5, 5])

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset, labels)):
        print('-' * 15, '>', f'Fold {fold}', '<', '-' * 15)

        path_model = './Kfold_models/fold{}/model.pkl'.format(fold)

        _, X_test = dataset[train_idx], dataset[test_idx]
        _, y_test = labels[train_idx], labels[test_idx]
        test_set = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset=test_set, batch_size=config.batch_size, shuffle=False)

        print('train_set: ', len(train_idx))
        print('test_set: ', len(test_idx))

        model = Transformer(config)
        model = model.to(config.device)
        model.load_state_dict(torch.load(path_model), strict=True)

        accuracy, average_sensitivity, average_specificity, con_mat = test(model, test_loader, config)

        ACC += accuracy
        Sens += average_sensitivity
        Spec += average_specificity

        Confusion_mat += con_mat

        del model

    ACC /= config.num_fold
    Sens /= config.num_fold
    Spec /= config.num_fold

    class_wise_result = class_wise_evaluate(Confusion_mat)

    return ACC, Sens, Spec, Confusion_mat, class_wise_result


if __name__ == '__main__':
    config = Config()
    path = Path()

    ACC, Sens, Spec, Confusion_mat, class_wise_result = evaluate(config=config, path=path)

    print('ACC: ', ACC)
    print('Sens: ', Sens)
    print('Spec: ', Spec)
    print('confusion_mat:')
    print(Confusion_mat)
    print('class_wise_result: ')
    print(class_wise_result)
