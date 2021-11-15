import numpy as np
from torchvision import datasets, transforms

def iid(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    
    for i in range(num_users):
        
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    
    return dict_users

def noniid(dataset, args):
    """
    Sample non-I.I.D client data from dataset
    -> Different clients can hold vastly different amounts of data
    :param dataset:
    :param num_users:
    :return:
    """
    num_dataset = len(dataset)
    idx = np.arange(num_dataset)
    dict_users = {i: list() for i in range(args.num_users)}
    
    min_num = 100
    max_num = 700

    random_num_size = np.random.randint(min_num, max_num+1, size=args.num_users)
    print(f"Total number of datasets owned by clients : {sum(random_num_size)}")

    # total dataset should be larger or equal to sum of splitted dataset.
    assert num_dataset >= sum(random_num_size)

    # divide and assign
    for i, rand_num in enumerate(random_num_size):

        rand_set = set(np.random.choice(idx, rand_num, replace=False))
        idx = list(set(idx) - rand_set)
        dict_users[i] = rand_set

    return dict_users
