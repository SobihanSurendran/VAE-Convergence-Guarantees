import numpy as np
import pickle
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from IWAE_Model import IWAE
from outils import load_dataset


model_params = {
    'dataset': 'CelebA',                    # Dataset used for training
    'latent_size': 100,                     # Dimensionality of the latent space
    'K': 5,                                 # Number of variational samples used for the IWAE
    'soft_cipping_activation': True,        # Soft clipping activation enabled or not
    's': 5.0,                               # Sharpness of the transition in the Soft clipping (see paper)
    'batch_size': 256                       # Size of training batches
}


trainer_params = {
    'optimizer': {
        'type': 'Adam',                     # Type of optimizer used
        'parameters': {
            'lr': 0.001,                    # Learning rate of the optimizer
            'betas': (0.9, 0.999),          # Momentum coefficients for Adam
            'weight_decay': 0,              # Weight decay value (L2 regularization)
            'eps': 1e-8,                    # Regularization term to prevent division by zero
            'amsgrad': False                # AMSGrad (a variant of Adam) enabled or not
        }
    },
    'gradient_clip': False,                 # Gradient clipping enabled or not
    'gradient_clip_value': 1.0,             # Value for gradient clipping if enabled
    'num_epochs': 100,                      # Number of training epochs
    'T': 1,                                 # Number of independent runs
    'file_name': 'IWAE_CelebA.pkl'          # Loss and gradient data file name
}


if __name__ == "__main__":

    #torch.manual_seed(0)

    train_loader, test_loader = load_dataset(model_params)

    # Define the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    losses_dict_list = []

    for i in range(trainer_params['T']):

        # Initialize the model
        model = IWAE(model_params).to(device)

        # Create a dictionary to store loss values for the model
        losses_dict = {'train': [], 'test': [], 'grad_norm':[]}

        # Define the optimizer
        optimizer_type = trainer_params['optimizer']['type']
        optimizer_parameters = trainer_params['optimizer']['parameters']
        optimizer_class = getattr(optim, optimizer_type)
        optimizer = optimizer_class(model.parameters(), **optimizer_parameters)

        for epoch in range(trainer_params['num_epochs']):
            model.train()
            train_loss = 0

            # Initialize a variable to accumulate the squared gradient norms for this epoch
            epoch_squared_gradient_norm = 0

            lr = optimizer_parameters['lr'] / np.sqrt(epoch + 1)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            for batch_idx, batch in enumerate(train_loader):
                data, labels = batch if isinstance(batch, list) else (batch, None)
                data = data.to(device)
                optimizer.zero_grad()

                x_hat, mu, logvar, z = model(data)
                loss = model.loss(x_hat, data, mu, logvar, z)

                loss.backward()

                # Clip gradients
                if trainer_params['gradient_clip']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), trainer_params['gradient_clip_value'])

                # Accumulate the squared gradient norms for this batch
                epoch_squared_gradient_norm += sum(p.grad.norm().item() ** 2 for p in model.parameters())

                optimizer.step()
                train_loss += loss.item()

            # Calculate the average training loss for the epoch and append it to the corresponding model's loss list
            avg_train_loss = train_loss / len(train_loader.dataset)
            losses_dict['train'].append(avg_train_loss)

            # Append the squared gradient norm for this epoch to the list
            losses_dict['grad_norm'].append(epoch_squared_gradient_norm / (len(train_loader.dataset)*sum(p.numel() for p in model.parameters())))
            print('Epoch: [{}/{}], Training Loss: {:.3f}'.format(epoch+1, trainer_params['num_epochs'], avg_train_loss))

            # Evaluate the model on the test dataset
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch in test_loader:
                    data, labels = batch if isinstance(batch, list) else (batch, None)
                    data = data.to(device)
                    x_hat, mu, logvar, z = model(data)
                    loss = model.loss(x_hat, data, mu, logvar, z)
                    test_loss += loss.item()

            # Calculate the average test loss for the epoch and append it to the corresponding model's loss list
            avg_test_loss = test_loss / len(test_loader.dataset)
            losses_dict['test'].append(avg_test_loss)

            print('Epoch: [{}/{}], Test Loss: {:.3f}'.format(epoch+1, trainer_params['num_epochs'], avg_test_loss))

        losses_dict_list.append(losses_dict)

        # Saving the dictionary
        with open(trainer_params['file_name'], 'wb') as f:
            pickle.dump(losses_dict_list, f)
