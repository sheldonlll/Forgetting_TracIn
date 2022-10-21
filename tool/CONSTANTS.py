from torchvision.transforms import transforms

lr = 1e-3
test_batch_size = 256
train_batch_size = 1200
epoches = 42

checking_points_cpts1 = "./cpts1/"
checking_points_cpts2 = "./cpts2/"

data_transform = {
"train": transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

"val": transforms.Compose([transforms.Resize((32, 32)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}

acc_detail_per_epoch_file_path = "C:\\Users\\Sherlock\\Desktop\\pycodes\\Forgetting_TracIn\\acc_per_epoch_detail_lst.txt"
forgetting_score_results = "C:\\Users\\Sherlock\\Desktop\\pycodes\\Forgetting_TracIn\\forgetting_score_results.txt"
TracIn_results = "C:\\Users\\Sherlock\\Desktop\\pycodes\\Forgetting_TracIn\\Tracin_results"
TracIn_original_train_indexes_original_test_indexes = "C:\\Users\\Sherlock\\Desktop\\pycodes\\Forgetting_TracIn\\TracIn_original_train_indexes_original_test_indexes"