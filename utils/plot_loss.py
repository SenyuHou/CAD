import matplotlib.pyplot as plt
import os

def plot_and_save_losses(losses, log_name, log_dir='./losses_pic'):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    plt.figure(figsize=(10, 5))  
    plt.plot(losses, label='Training Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True) 

    plt.savefig(f'{log_dir}/{log_name}.png')
    plt.close()  