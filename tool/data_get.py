import torchvision.transforms as transforms
import torchvision
import torch
def dataset_get():
    pass

def dataset_category_get(category_num, train_size, test_size, train_dataloader, test_dataloader):
    data_transform = {
        "train": transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),# converts images loaded by Pillow into PyTorch tensors.
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    } # transform incoming images into a Pytorch Tensor
    
    
    test_set = torchvision.datasets.CIFAR10(root = "datasets", train = False, download = False, transform = data_transform["val"])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = test_size, shuffle = False, num_workers = 0)

    train_set = torchvision.datasets.CIFAR10(root = "datasets", train = False, download = False, transform = data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = train_size, shuffle = False, num_workers = 0)
    
    train_data_iter = iter(train_loader)
    train_image, train_label = train_data_iter.next()

    img_all_train = torch.zeros(train_size, 3, 224, 224) # 存放train_image中所有标签是参数category_num的图片作为训练数据集
    train_image_num = 0 # img_all_train数组的当前数量/下标


    original_train_indexes = []
    original_test_indexes = []

    for i in range(min(train_size, len(train_dataloader))):
        if (train_label[i] == category_num):
            img_all_train[train_image_num] = train_image[i]
            train_image_num += 1
            original_train_indexes.append(i)
        if train_image_num == train_size:
            break
    


    test_data_iter = iter(test_loader)
    test_image, test_label = test_data_iter.next()

    img_all_test = torch.zeros(test_size, 3, 224, 224) # 存放test_image中所有标签是参数category_num的图片作为训练数据集
    test_image_num = 0 # img_all_train数组的当前数量/下标

    for i in range(min(test_size, len(test_dataloader))):
        if (test_label[i] == category_num):
            img_all_test[test_image_num] = test_image[i]
            test_image_num += 1
            original_test_indexes.append(i)
        if test_image_num == test_size:
            break
    
    return img_all_train, img_all_test, original_train_indexes, original_test_indexes # shape: (train_size, 3, 224, 224), (test_size, 3, 224, 224)
    
    
