import torch



def dataset_get(train_dataloader, test_dataloader, train_size = 0, test_size = 0):
    if train_size == 0:
        train_size = len(train_dataloader)
    if test_size == 0:
        test_size = len(test_dataloader)

    train_data_iter = iter(train_dataloader)
    train_image, train_label = train_data_iter.next()

    img_all_train = torch.zeros(train_size, 3, 224, 224) # 存放train_image中所有标签是参数category_num的图片作为训练数据集
    train_image_num = 0 # img_all_train数组的当前数量/下标，最多500张

    for i in range(min(train_size, len(train_dataloader))):
        img_all_train[train_image_num] = train_image[i]
        train_image_num += 1
        if train_image_num == train_size:
            break
    

    test_data_iter = iter(test_dataloader)
    test_image, test_label = test_data_iter.next()

    img_all_test = torch.zeros(test_size, 3, 224, 224) # 存放test_image中所有标签是参数category_num的图片作为训练数据集
    test_image_num = 0 # img_all_train数组的当前数量/下标，最多100张

    for i in range(min(test_size, len(test_dataloader))):
        img_all_test[test_image_num] = test_image[i]
        test_image_num += 1
        if test_image_num == test_size:
            break
    
    return img_all_train, img_all_test # shape: (500, 3, 224, 224), (100, 3, 224, 224)


def dataset_category_get(category_num, train_size, test_size, train_dataloader, test_dataloader):

    train_data_iter = iter(train_dataloader)
    train_image, train_label = train_data_iter.next()

    img_all_train = torch.zeros(train_size, 3, 224, 224) # 存放train_image中所有标签是参数category_num的图片作为训练数据集
    train_image_num = 0 # img_all_train数组的当前数量/下标，最多500张

    for i in range(min(train_size, len(train_dataloader))):
        if (train_label[i] == category_num):
            img_all_train[train_image_num] = train_image[i]
            train_image_num += 1
        if train_image_num == train_size:
            break
    

    test_data_iter = iter(test_dataloader)
    test_image, test_label = test_data_iter.next()

    img_all_test = torch.zeros(test_size, 3, 224, 224) # 存放test_image中所有标签是参数category_num的图片作为训练数据集
    test_image_num = 0 # img_all_train数组的当前数量/下标，最多100张

    for i in range(min(test_size, len(test_dataloader))):
        if (test_label[i] == category_num):
            img_all_test[test_image_num] = test_image[i]
            test_image_num += 1
        if test_image_num == test_size:
            break
    
    return img_all_train, img_all_test # shape: (500, 3, 224, 224), (100, 3, 224, 224)