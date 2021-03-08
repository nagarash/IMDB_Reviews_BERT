import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

def sigmoid(x):                                        
    return 1 / (1 + np.exp(-x))

def get_test_dataloader(data_dir, batch_size):
    print("Get test data loader.")
    
    
    # read test tensors from pickled object
    with open("{}/test_tensors.pkl".format(data_dir), "rb") as f:
            test_seq, test_mask, test_y = pickle.load(f)
    test_y_tensor = torch.tensor(np.array(test_y).reshape(-1,1)).float()

    test_dataset = TensorDataset(test_seq, test_mask, test_y_tensor)
    test_sampler =  SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)
    
    return test_y, test_dataloader

def evaluate(model, data_dir, batch_size, optimizer, loss_fn):
    """
    Function to evaluate model performance on the test dataset.
    model: trained pytorch model object
    optimizer: the optimizer used during prediction runs (should be same as in training phase)
    loss_func: the loss function used during prediction runs (should be as same as in training phase)
    test_dataloader: dataloader for test dataset
    """
       
    # check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # push the model to GPU
    model.to(device)
    
    # set model for evaluation 
    model.eval()
    
    # get test dataloader
    test_y, test_dataloader = get_test_dataloader(data_dir, batch_size)
    
    y_pred = []
    y_prob = []
    steps = []
    test_losses = []
    
    step = 0
    test_loss = 0

    for step_num, batch_data in enumerate(test_dataloader):
        token_ids, masks, labels = tuple(t.to(device) for t in batch_data)
        
        # evaluate
        probas = model(token_ids, masks)
        numpy_probas = probas.cpu().detach().numpy()
        
        # calculate loss
        batch_loss = loss_fn(probas, labels)
        test_loss += batch_loss.item()
        steps.append(step)
        
        step += 1
        print("{0}/{1} loss: {2} ".format(step_num,
                                          len(test_y) / batch_size, 
                                          test_loss / (step_num + 1)), end="\r", flush=True)
        
        test_losses.append(test_loss / (step_num + 1))
        # convert probab to binary 
        y_prob += list(numpy_probas)
        y_pred += list(sigmoid(numpy_probas[:, 0]) > 0.5)
    
    return steps, test_losses, y_prob, y_pred, test_y