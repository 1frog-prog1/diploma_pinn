import json 
import torch

def save_eval_losses(eval_losses, filename):
    with open(filename, 'w') as f:
        json.dump(eval_losses, f)

def save_u_results(T, X, U, filename):
    with open(filename, 'w') as f:
        json.dump({
            "t_map": T.tolist(), "x_map": X.tolist(), "u_map": U.tolist()
        }, f)

def save_model(model, filename):
    """Save the model's state dictionary to a file."""
    torch.save(model.cpu().state_dict(), filename)

def load_model(model, filename):
    """Load the model's state dictionary from a file."""
    model.load_state_dict(torch.load(filename))