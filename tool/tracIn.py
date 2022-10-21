from glob import glob
import time
import os
import torch
from torch import nn

from tool.model import resnet34
from tool.data_get import dataset_category_get, dataset_get
from tool.CONSTANTS import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_train_indexes, original_test_indexes = [], []


def get_gradient(grads, model):
    return [grad for grad, (n, p) in zip(grads, model.named_parameters())]


def tracin_get(a, b):
    return sum([torch.dot(at.flatten(), bt.flatten()) for at, bt in zip(a, b)])


def calculate_tracin(category_num, path, file_name, learning_rate, batch_size, train_dataloader, test_dataloader):
    model_weight_path = path + file_name
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    net = resnet34()
    checkpoint = torch.load(model_weight_path, map_location = device)
    net.load_state_dict(checkpoint["state_dict"])
    net.to(device = device)
    print(f"net: {net}")

    loss_fn = nn.CrossEntropyLoss()

    global original_train_indexes
    global original_test_indexes
    
    img_all_train, img_all_test, original_train_indexes, original_test_indexes = dataset_category_get(category_num = category_num, train_size = len(train_dataloader), test_size = len(test_dataloader), train_dataloader = train_dataloader, test_dataloader = test_dataloader)

    train_size = len(train_dataloader)
    test_size = len(test_dataloader)

    img_all_train = img_all_train.view(train_size, 1, 3, 224, 224)
    img_all_test = img_all_test.view(test_size, 1, 3, 224, 224)

    label_train = torch.zeros(1).long()
    logits_train = net(img_all_train[0].to(device))
    loss_train = loss_fn(logits_train, label_train.to(device))
    grad_z_train = torch.autograd.grad(loss_train, net.parameters())
    grad_z_train = get_gradient(grads = grad_z_train, model = net)

    score_list = []
    time_start = time.perf_counter()

    for i in range(test_size):
        label_test = torch.zeros(1).long()
        label_test[0] = 0
        logits_test = net(img_all_test[i].to(device))
        loss_test = loss_fn(logits_test, label_test.to(device))
        grad_z_test = torch.autograd.grad(loss_test, net.parameters())
        grad_z_test = get_gradient(grads = grad_z_test, model = net)

        score = tracin_get(grad_z_test, grad_z_train)
        score_list.append(float(score) * learning_rate / batch_size)
    
    print("%f s" % (time.perf_counter() - time_start))
    print(score_list)
    return score_list


def process_train_data_via_tracIn(lr, test_batch_size, checking_points_path, train_dataloader, test_dataloader):
    score_final = []
    learning_rate = lr
    batch_size = test_batch_size
    path = checking_points_path

    for category_num in range(10):
        for file_name in os.listdir(path):
            current_epoch_score_list = calculate_tracin(category_num = category_num, path = path, file_name = file_name, learning_rate = learning_rate, batch_size = batch_size, train_dataloader = train_dataloader, test_dataloader = test_dataloader)
            if len(score_final) == 0:
                score_final = current_epoch_score_list
            else:
                temp_list = []
                for x, y in zip(current_epoch_score_list, score_final):
                    temp_list.append(x + y)
                score_final = temp_list
        print(score_final)

        with open(TracIn_results + "_" + str(category_num) + ".txt", "w+") as f:
            cur_data = str(score_final)
            f.write(cur_data)

        with open(TracIn_original_train_indexes_original_test_indexes + "_" + str(category_num) + ".txt", "w+") as f:
            cur_data = str(original_train_indexes) + "\n" + str(original_test_indexes)
            f.write(cur_data)

