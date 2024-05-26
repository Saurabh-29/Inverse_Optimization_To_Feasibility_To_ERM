import wandb

sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'loss',
        'goal': 'minimize'
    },
    'optimizer': {
        'learning_rate': {
            'values': [0.01, 0.001, 0.0001]
        },
        'batch_size': {
            'values': [32, 64, 128]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="your_project_name")

# Define a function to call the training script
def main():
    import os
    os.system("python train.py")

wandb.agent(sweep_id, function=main, count=100)