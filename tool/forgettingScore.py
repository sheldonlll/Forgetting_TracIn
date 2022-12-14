import os
import json
from itertools import chain

from tool import CONSTANTS


all_epoch_detail_dict = {}


def prepare_data(data):
    len_per_epoch = 0
    
    for i in range(len(data)):
        epoch_idx, current_epoch_accuracy_detail_str = data[i].split(" = ")[0], data[i].split(" = ")[-1]
        idx = current_epoch_accuracy_detail_str.find("[", 1)
        current_epoch_accuracy_detail_seperate_by_epoch_lst = json.loads("[" + current_epoch_accuracy_detail_str[:idx - 2] + "], " + current_epoch_accuracy_detail_str[idx:])
        
        current_epoch_accuracy_detail_lst = list(chain(*current_epoch_accuracy_detail_seperate_by_epoch_lst))
        len_per_epoch = len(current_epoch_accuracy_detail_lst)
        all_epoch_detail_dict[epoch_idx] = current_epoch_accuracy_detail_lst

    return len_per_epoch


def Algorithm_1_Computing_forgetting_statistics(to_shuffle = False, len_per_epoch = 0):
    forgetting_event_happend_state = [False for _ in range(len_per_epoch)]
    learning_event_happend_state = [False for _ in range(len_per_epoch)]
    first_learning_event_happened_state = [-1 for _ in range(len_per_epoch)]
    first_forgetting_event_happend_state = [-1 for _ in range(len_per_epoch)]

    tot_forgetting_score = [0 for _ in range(len_per_epoch)]
    unforgettable_examples = []
    forgettable_examples = []
    global all_epoch_detail_dict 
    all_epoch_detail_dict = {k: v for k, v in sorted(all_epoch_detail_dict.items(), key=lambda item: item[0].split("_")[-1])}
    for current_batch_idx in range(len(all_epoch_detail_dict)): # while not training done, across all epoches
        next_batch_idx = min(current_batch_idx + 1, len(all_epoch_detail_dict) - 1)
        for i in range(len_per_epoch):
            current_acc = all_epoch_detail_dict["epoch_" + str(current_batch_idx)][i]
            next_acc = all_epoch_detail_dict["epoch_" + str(next_batch_idx)][i]
            
            if first_learning_event_happened_state[i] == -1 and current_acc < next_acc:
                first_learning_event_happened_state[i] = current_batch_idx
            if first_forgetting_event_happend_state[i] == -1 and learning_event_happend_state[i] and current_acc < next_acc:
                first_forgetting_event_happend_state[i] = current_batch_idx
            
            if current_acc > next_acc:
                tot_forgetting_score[i] -= 2
            elif current_acc == next_acc and current_acc == 0:
                tot_forgetting_score[i] -= 1
            elif current_acc == next_acc and current_acc == 1:
                tot_forgetting_score[i] += 1
            elif current_acc < next_acc:
                tot_forgetting_score[i] += 2
        
            forgetting_event_happend_state[i] = True if current_acc > next_acc else forgetting_event_happend_state[i]
            learning_event_happend_state[i] = True if current_acc < next_acc else learning_event_happend_state[i]
            

    for i in range(len_per_epoch):
        if forgetting_event_happend_state[i] == False and learning_event_happend_state[i] == True:
            unforgettable_examples.append(i)
        if forgetting_event_happend_state[i]:
            forgettable_examples.append(i)
    
    return tot_forgetting_score, unforgettable_examples, forgettable_examples, first_learning_event_happened_state, first_forgetting_event_happend_state


def process_train_data_via_forgetting(train_dataloader):
    file_data = None
    file_path = input("?????????????????????epoch???, ???????????????????????????????????????????????????(???????????????epoch_0 = [...] epoch_1 = [...])?????????")
    if os.path.exists(file_path) == False:
        file_path = CONSTANTS.acc_detail_per_epoch_file_path
        with open(file_path, "r") as f:
            file_data = f.readlines()

        len_per_epoch = prepare_data(file_data)
        tot_forgetting_score, unforgettable_examples, forgettable_examples, first_learning_event_happened_state, first_forgetting_event_happend_state = Algorithm_1_Computing_forgetting_statistics(to_shuffle = False, len_per_epoch = len_per_epoch) #??????forgettingscore
        with open(CONSTANTS.forgetting_score_results, "w+") as f:
            data = "tot_forgetting_score:\n" + str(tot_forgetting_score) + "\nunforgettable_examples:\n"+ str(unforgettable_examples) + "\nforgettable_examples:\n" + str(forgettable_examples) + "\nfirst_learning_event_happened_state:\n" + str(first_learning_event_happened_state) + "\nfirst_forgetting_event_happend_state:\n" + str(first_forgetting_event_happend_state)
            f.write(data)
    else:
        with open(file_path, "r") as f:
            file_data = f.readlines()
        len_per_epoch = int(input("len_per_epoch: "))
        Algorithm_1_Computing_forgetting_statistics(to_shuffle = False, len_per_epoch = len_per_epoch)