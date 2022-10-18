import os
import datetime

import torch
import numpy as np
from matplotlib import pyplot as plt
from torchvision import datasets
from torch import optim

from tool.CONSTANTS import * # lr, test_batch_size, train_batch_size, epoches, checking_points_cpts1, checking_points_cpts2, data_transform, acc_detail_per_epoch_file_path
from tool.model import resnet34

from tool.forgettingScore import process_train_data_via_forgetting
from tool.tracIn import process_train_data_via_tracIn
from tool.maintool import * # select_train_dataloader_via_forgetting_tracin, compare_result_if_filtered_or_not, find_correlation_between_Forgetting_and_TracIn


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"


def load_data(data_trasform = data_transform, test_batch_size = 32, train_batch_size = 32, download = True, shuffle = True):
    cifar10_train = datasets.CIFAR10(root = "datasets", train = True, download = download, transform = data_transform["train"])
    cifar10_test = datasets.CIFAR10(root = "datasets", train = False, download = download, transform = data_transform["val"])
    print(f"cifar10_train size: {len(cifar10_train)} \t cifar10_test_size: {len(cifar10_test)}")
    kwargs = {'num_workers': 6, 'pin_memory': True} if torch.cuda.is_available() else {}
    cifar10_train_dataloader = torch.utils.data.DataLoader(cifar10_train, batch_size = train_batch_size, shuffle = shuffle, num_workers = kwargs["num_workers"], pin_memory = kwargs["pin_memory"])
    cifar10_test_dataloader = torch.utils.data.DataLoader(cifar10_test, batch_size = test_batch_size, shuffle = shuffle, num_workers = kwargs["num_workers"], pin_memory = kwargs["pin_memory"])
    return cifar10_train_dataloader, cifar10_test_dataloader


def draw_accuracy_loss_line(tot_epoch, loss_lst, acc_lst):
    #画loss，acc line
    x_epoches = np.arange(tot_epoch)
    loss_lst = np.array(loss_lst)
    acc_lst = np.array(acc_lst)
    plt.plot(x_epoches, loss_lst, label = "loss line")
    plt.plot(x_epoches, acc_lst, label = "accuracy line")
    plt.title("loss, accuracy -- line")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()


def save_accuracy_per_epoch_detail(accuracy_detail_dict, file_path):
    #保存预测结果
    with open(file_path, "w+") as f:
        cur_data = ""
        for key in sorted(accuracy_detail_dict.keys()):
            cur_data += "epoch_" + str(key) + " = " + str(accuracy_detail_dict[key]) + "\n"
        f.write(cur_data)


def train_predict(train_dataloader, test_dataloader, lr, epoches, save_checking_points = True, checking_points_path = "./cpts/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet34.to(device)
    criteon = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.paremeters(), lr = lr)

    loss_lst, acc_lst = [], []
    accuracy_detail_dict = {}
    tot_epoch = 0

    for epoch in range(epoches):
        model.train()
        loss = torch.tensor(-1.0)
        lossMIN = 0x3fff
        launchTimestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for _, (x, label) in enumerate(train_dataloader):
            x, label = x.to(device), label.to(device)

            try:
                logits = model(x)
                loss = criteon(logits, label)
                lossMIN = min(lossMIN, loss)
                optimizer.zero_grad()
                loss.backword()
                optimizer.step()
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception
        
        loss_lst.append(lossMIN.cpu().detach().numpy())
        print(f"launchTimestamp: {launchTimestamp} epoch: {epoch + 1}, current epoch min loss: {lossMIN.item()}")

        model.eval()
        with torch.no_grad():
            tot_correct = 0
            tot_num = 0

            for x, label in test_dataloader:
                x, label = x.to(device), label.to(device)
                logits = model(x) # [batchsize, 10]
                pred = logits.argmax(dim = 1)
                result = torch.eq(pred, label)
                
                '''
                当前batch没看过将tot_epoch作为key, 第一个batch的预测结果作为value
                否则将tot_epoch作为key, append当前batch的预测结果
                '''
                if accuracy_detail_dict.get(tot_epoch) == None:
                    accuracy_detail_dict[tot_epoch] = result.int().tolist()
                else:
                    accuracy_detail_dict[tot_epoch].append(result.int().tolist())
                
                tot_correct += result.float().sum().item()
                tot_num += x.shape(0) # [batchsize, 10]

            accuracy = tot_correct / tot_num
            print(f"launchTimestamp: {launchTimestamp} epoch: {epoch + 1}, accuracy: {accuracy}")

        acc_lst.append(accuracy)
        torch.save({"epoch": epoch + 1, "state_dict": model.state_dict(), "min_loss": lossMIN, "optimizer": optimizer.state_dict()}, 
                    checking_points_path + "-" + str("%.4f" % lossMIN) + ".pth.tar")

        tot_epoch += 1

    draw_accuracy_loss_line(tot_epoch = tot_epoch, loss_lst = loss_lst, acc_lst = acc_lst)
    save_accuracy_per_epoch_detail(accuracy_detail_dict = accuracy_detail_dict, file_path = acc_detail_per_epoch_file_path)

    return model


def main():
    torch.cuda.empty_cache()

    train_dataloader, test_dataloader = load_data(test_batch_size = test_batch_size, train_batch_size = train_batch_size, download = True, shuffle = True)
    
    result_no_filter_train_data = train_predict(train_dataloader, test_dataloader, lr = lr, epoches = epoches,save_checking_points = True, checking_points_path = checking_points_cpts1)
    
    # TODO
    train_dataloader_via_forgetting = process_train_data_via_forgetting(train_dataloader)
    train_dataloader_via_tracIn = process_train_data_via_tracIn(lr = lr, test_batch_size = test_batch_size, checking_points_path = checking_points_cpts1, train_dataloader = train_dataloader, test_dataloader = test_dataloader)
    train_dataloader = select_train_dataloader_via_forgetting_tracin(train_dataloader_via_forgetting, train_dataloader_via_tracIn)
    
    result_filtered_train_data = train_predict(train_dataloader, test_dataloader, lr = lr, epoches = epoches, save_checking_points = True, checking_points_path = checking_points_cpts2)
    
    compare_result_if_filtered_or_not(result_no_filter_train_data, result_filtered_train_data)

    find_correlation_between_Forgetting_and_TracIn(train_dataloader_via_forgetting, train_dataloader_via_tracIn)


if __name__ == "__main__":
    main()