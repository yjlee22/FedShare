import numpy as np

def uniform_distribute(dataset, args): 
    globally_shared_data_idx = []
    
    idxs = np.arange(len(dataset))
    
    if args.dataset == "mnist":
        labels = dataset.targets.numpy()
    elif args.dataset == "cifar":
        labels = np.array(dataset.targets)
    else:
        exit('Error: unrecognized dataset')
    
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]

    idxs = idxs_labels[0]
    labels = idxs_labels[1]
    
    for i in range(args.num_classes):
        specific_class = np.extract(labels == i, idxs)
        globally_shared_data = np.random.choice(specific_class, int(args.alpha * args.classwise), replace=False)
        
        globally_shared_data_idx = globally_shared_data_idx + list(globally_shared_data)
    
    return globally_shared_data_idx

def train_dg_split(dataset, args): 
    dg_idx = []
    train_idx = []
    idxs = np.arange(len(dataset))

    if args.dataset == "mnist":
        labels = dataset.targets.numpy()
    elif args.dataset == "cifar":
        labels = np.array(dataset.targets)
    else:
        exit('Error: unrecognized dataset')
    
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    
    idxs = idxs_labels[0]
    labels = idxs_labels[1]
    
    for i in range(args.num_classes):
        specific_class = np.extract(labels == i, idxs)
        
        dg = np.random.choice(specific_class, args.classwise, replace=False)
        
        train_tmp = set(specific_class)-set(dg)
        
        dg_idx = dg_idx + list(dg)
        
        train_idx = train_idx + list(train_tmp)
    
    return dg_idx, train_idx    