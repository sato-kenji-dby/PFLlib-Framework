# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import os
import ujson
import numpy as np
import gc
from sklearn.model_selection import train_test_split

batch_size = 10
train_ratio = 0.75 # merge original training set and test set, then split it manually. 
alpha = 0.5 # for Dirichlet distribution. 100 for exdir

import os
import ujson
import shutil

def check(config_path, train_path, test_path, num_clients, niid=False, 
        balance=True, partition=None):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
            config['non_iid'] == niid and \
            config['balance'] == balance and \
            config['partition'] == partition and \
            config['alpha'] == alpha and \
            config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False

def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=None, target_count=3000, client0_label_num=None):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data
    # guarantee that each client must have at least one batch of data for testing. 
    least_samples = int(min(batch_size / (1-train_ratio), len(dataset_label) / num_clients / 2))

    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            if len(selected_clients) == 0:
                break
            selected_clients = selected_clients[:int(np.ceil((num_clients/num_classes)*class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
            else:
                num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
            num_samples.append(num_all_samples-sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1
            

    elif partition == "dir":
        '''
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print(f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]

        # *** REVISION STARTS HERE ***
        # Number of labels client 0 should receive, unrelated to original Dirichlet distribution

        # Choose client0_label_num distinct labels from the full range of labels
        # If client0_label_num > K, we can handle by taking min or raising an error.
        client0_label_num = min(client0_label_num, K)
        all_labels = np.arange(K)
        chosen_labels = np.random.choice(all_labels, size=client0_label_num, replace=False)

        # Gather all data points belonging to these chosen labels
        chosen_indices = np.where(np.isin(dataset_label, chosen_labels))[0]

        # Assign these indices to client 0, overriding previous assignment
        dataidx_map[0] = chosen_indices.tolist()
        # *** REVISION ENDS HERE ***
        '''
        dataidx_map = {j: [] for j in range(num_clients)}
        K = num_classes
        N = len(dataset_label)

        if num_clients == 2:
            # 特殊处理：当有2个客户端时
            print("使用特定策略分配数据给2个客户端。")

            client1_label_allocation = {
                5: 529,
                6: 597,
                7: 643,
                8: 598,
                9: 633
            }

            # 1. 为客户端0分配标签 0 到 client0_label_num - 1，并平均分配每个标签的数据点
            for label in range(client0_label_num):
                idx_label = np.where(dataset_label == label)[0]
                np.random.shuffle(idx_label)  # 随机打乱索引
                num_samples = len(idx_label) // num_clients  # 平均分配

                # 若数据不足，则通过重复已有样本补足
                if num_samples == 0:
                    # 如果一个标签的数据连平均分配1个都不够，至少取1个，然后重复
                    num_samples = 1
                if len(idx_label) < num_samples:
                    # 使用重复补足
                    selected_indices = np.random.choice(idx_label, size=num_samples, replace=True)
                else:
                    selected_indices = idx_label[:num_samples]

                dataidx_map[0].extend(selected_indices.tolist())

            # 2. 为客户端1分配标签编号 >= K/2，并手动指定每个标签的数据点数量
            min_label_client1 = K // 2
            for label in range(min_label_client1, K):
                if label not in client1_label_allocation:
                   raise ValueError(f"未为客户端1的标签 {label} 指定数据点数量。请在 client1_label_allocation 中添加该标签的分配数量。")

                allocated_count = client1_label_allocation[label]
                idx_label = np.where(dataset_label == label)[0]
                np.random.shuffle(idx_label)  # 随机打乱索引

                # 若数据不足，通过重复补足
                if len(idx_label) < allocated_count:
                    selected_indices = np.random.choice(idx_label, size=allocated_count, replace=True)
                else:
                    selected_indices = idx_label[:allocated_count]

                dataidx_map[1].extend(selected_indices.tolist())

            # 3. 确保每个客户端的数据量不超过 target_count（根据需求调整）
            for client in range(num_clients):
                if len(dataidx_map[client]) > target_count:
                    selected_indices = np.random.choice(dataidx_map[client], size=target_count, replace=False)
                    dataidx_map[client] = selected_indices.tolist()
                elif len(dataidx_map[client]) < target_count:
                    # 数据不足则重复已有的数据
                    needed = target_count - len(dataidx_map[client])
                    additional = np.random.choice(dataidx_map[client], size=needed, replace=True)
                    dataidx_map[client].extend(additional.tolist())

        else:
            # 原有的分配策略，适用于 num_clients != 2 (逻辑不变，如果需要同样可重复补足)
            try_cnt = 1
            while True:
                if try_cnt > 1:
                    print(f'第 {try_cnt} 次尝试分配数据。')

                idx_batch = [[] for _ in range(num_clients)]
                for k in range(K):
                    idx_k = np.where(dataset_label == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                    
                    # 确保每个客户端的数据量不超过 target_count
                    proportions = np.array([p * (len(idx_j) < target_count) for p, idx_j in zip(proportions, idx_batch)])
                    if proportions.sum() == 0:
                        proportions = np.ones(num_clients) / num_clients  # 防止除以零
                    else:
                        proportions = proportions / proportions.sum()
                    
                    split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_pieces = np.split(idx_k, split_points)
                    
                    for i, piece in enumerate(idx_pieces):
                        idx_batch[i].extend(piece.tolist())
                
                min_size = min(len(idx_j) for idx_j in idx_batch)
                if min_size >= least_samples:
                    break
                try_cnt += 1
            
            # 初步分配完成
            for j in range(num_clients):
                dataidx_map[j] = idx_batch[j]
            
            # *** 归一化处理开始 ***
            # 1. 为客户端0选择指定数量的标签
            all_labels = np.arange(K)
            chosen_labels = np.random.choice(all_labels, size=client0_label_num, replace=True)
            
            # 2. 从选择的标签中获取所有对应的索引
            chosen_indices = np.where(np.isin(dataset_label, chosen_labels))[0]
            
            # 3. 随机选取 target_count 个样本分配给客户端0，不够则重复
            if len(chosen_indices) < target_count:
                # 重复补足
                selected_client0_indices = np.random.choice(chosen_indices, size=target_count, replace=True)
            else:
                selected_client0_indices = np.random.choice(chosen_indices, size=target_count, replace=False)
            dataidx_map[0] = selected_client0_indices.tolist()
            
            # 4. 确保其他客户端也有 target_count 个样本，不够则重复
            for j in range(1, num_clients):
                client_indices = dataidx_map[j]
                if len(client_indices) < target_count:
                    needed = target_count - len(client_indices)
                    additional = np.random.choice(client_indices, size=needed, replace=True)
                    dataidx_map[j].extend(additional.tolist())
                else:
                    selected_indices = np.random.choice(client_indices, size=target_count, replace=False)
                    dataidx_map[j] = selected_indices.tolist()

        # 输出 dataidx_map 以检查分配结果
        print("数据分配结果 (dataidx_map):")
        for client, indices in dataidx_map.items():
            print(f"客户端 {client}: {len(indices)} 个样本")

                
    elif partition == 'exdir':
        r'''This strategy comes from https://arxiv.org/abs/2311.03154
        See details in https://github.com/TsingZ0/PFLlib/issues/139

        This version in PFLlib is slightly different from the original version 
        Some changes are as follows:
        n_nets -> num_clients, n_class -> num_classes
        '''
        C = class_per_client
        
        '''The first level: allocate labels to clients
        clientidx_map (dict, {label: clientidx}), e.g., C=2, num_clients=5, num_classes=10
            {0: [0, 1], 1: [1, 2], 2: [2, 3], 3: [3, 4], 4: [4, 5], 5: [5, 6], 6: [6, 7], 7: [7, 8], 8: [8, 9], 9: [9, 0]}
        '''
        min_size_per_label = 0
        # You can adjust the `min_require_size_per_label` to meet you requirements
        min_require_size_per_label = max(C * num_clients // num_classes // 2, 1)
        if min_require_size_per_label < 1:
            raise ValueError
        clientidx_map = {}
        while min_size_per_label < min_require_size_per_label:
            # initialize
            for k in range(num_classes):
                clientidx_map[k] = []
            # allocate
            for i in range(num_clients):
                labelidx = np.random.choice(range(num_classes), C, replace=False)
                for k in labelidx:
                    clientidx_map[k].append(i)
            min_size_per_label = min([len(clientidx_map[k]) for k in range(num_classes)])
        
        '''The second level: allocate data idx'''
        dataidx_map = {}
        y_train = dataset_label
        min_size = 0
        min_require_size = 10
        K = num_classes
        N = len(y_train)
        print("\n*****clientidx_map*****")
        print(clientidx_map)
        print("\n*****Number of clients per label*****")
        print([len(clientidx_map[i]) for i in range(len(clientidx_map))])

        # ensure per client' sampling size >= min_require_size (is set to 10 originally in [3])
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_clients)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                # Balance
                # Case 1 (original case in Dir): Balance the number of sample per client
                proportions = np.array([p * (len(idx_j) < N / num_clients and j in clientidx_map[k]) for j, (p, idx_j) in enumerate(zip(proportions, idx_batch))])
                # Case 2: Don't balance
                #proportions = np.array([p * (j in label_netidx_map[k]) for j, (p, idx_j) in enumerate(zip(proportions, idx_batch))])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # process the remainder samples
                '''Note: Process the remainder data samples (yipeng, 2023-11-14).
                There are some cases that the samples of class k are not allocated completely, i.e., proportions[-1] < len(idx_k)
                In these cases, the remainder data samples are assigned to the last client in `clientidx_map[k]`.
                '''
                if proportions[-1] != len(idx_k):
                    for w in range(clientidx_map[k][-1], num_clients-1):
                        proportions[w] = len(idx_k)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))] 
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            dataidx_map[j] = idx_batch[j]
    
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))
            

    del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic


def split_data(X, y):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[]}

    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_ratio, shuffle=True)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y
    # gc.collect()

    return train_data, test_data

def save_file(config_path, train_path, test_path, train_data, test_data, num_clients, 
                num_classes, statistic, niid=False, balance=True, partition=None):
    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid': niid, 
        'balance': balance, 
        'partition': partition, 
        'Size of samples for labels in clients': statistic, 
        'alpha': alpha, 
        'batch_size': batch_size, 
    }

    # gc.collect()
    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")
