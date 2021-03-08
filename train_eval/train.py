import os
import pickle
import torch
import argparse
import numpy as np
from torch.nn.utils import clip_grad_norm_
from IPython.display import clear_output
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import Adam

from model import BertBinaryClassifier
from evaluate import evaluate


def model_fn():
    """Load the PyTorch model"""
    print("Loading model.")

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained model
    model = BertBinaryClassifier()
    model = model.cuda()
    
    # push the model to GPU
    model.to(device)

    print("Done loading model.")
    return model

def get_train_dataloader(data_dir, batch_size):
    print("Get train and test data loader.")
        
    # read train tensors from pickled object
    with open("{}/train_tensors.pkl".format(data_dir), "rb") as f:
            train_seq, train_mask, train_y = pickle.load(f)
    train_y_tensor = torch.tensor(np.array(train_y).reshape(-1,1)).float()

    train_dataset = TensorDataset(train_seq, train_mask, train_y_tensor)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    
    return train_y, train_dataloader


def train(model, data_dir, epochs, batch_size, optimizer, loss_fn, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    batch_size.  - The size of training batch
    optimizer    - The optimizer to use during training.
    loss_fn      - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
      
    
    # get train dataloader
    train_y, train_dataloader = get_train_dataloader(data_dir, batch_size)
    
    steps = []
    losses = []
    step = 0
    for epoch_num in range(epochs):
        model.train()
        train_loss = 0
        for step_num, batch_data in enumerate(train_dataloader):
            # forward pass
            token_ids, masks, labels = tuple(t.to(device) for t in batch_data)
            probas = model(token_ids, masks)
            
            # calculate loss
            batch_loss = loss_fn(probas, labels)
            train_loss += batch_loss.item()

            # reset gradients, backward pass
            model.zero_grad()
            batch_loss.backward()

            # clip gradients
            clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            optimizer.step()

            clear_output(wait=True)
            print("Epoch:{}, steps: {}/{}, loss: {} ".format(epoch_num+1, 
                                                             step_num, 
                                                             len(train_y) / batch_size, 
                                                             train_loss /(step_num + 1)), end="\r", flush=True)
            losses.append(train_loss /(step_num + 1))
            steps.append(step)
            step += 1
    return model, steps, losses

if __name__ == '__main__':
    
    
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch_size', type=int, default=5, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=1)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)
    
    # initialize model
    model = model_fn()
    
    # set optimizer
    optimizer = Adam(model.parameters(), lr=3e-6)
    
    # set loss function
    loss_fn = torch.nn.BCEWithLogitsLoss().cuda()
    
    
    # Train the model.
    model, train_steps, train_loss = train(model, args.data_dir, args.epochs, args.batch_size, 
                                                 optimizer, loss_fn, device)


    # Save train losses
    loss_path = os.path.join(args.model_dir, 'train_losses.pt')
    with open(loss_path, 'wb') as f:
        pickle.dump([train_steps, train_loss], f)
    
    
    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pt')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
        
       
    # Evaluate on test set
    test_steps, test_loss, y_prob, y_pred, test_y = evaluate(model, args.data_dir, args.batch_size, 
                                                                      optimizer, loss_fn)
    
    # Save test losses and predictions
    loss_path = os.path.join(args.model_dir, 'test_losses.pt')
    with open(loss_path, 'wb') as f:
        pickle.dump([test_steps, test_loss, y_prob, y_pred, test_y], f)
    