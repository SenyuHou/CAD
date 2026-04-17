import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from utils.ema import EMA
import torch.optim as optim
from utils.learning import *
from utils.vit_wrapper import  vit_img_wrap
from utils.data_voc2007 import *
from utils.model_diffusion import Diffusion
from utils.ws_augmentation import *
from utils.log_config import setup_logger
from utils.plot_loss import plot_and_save_losses
from utils.knn_utils import *
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)

# Main training function (diffusion model, training, validation, test sets, model save path, command line arguments, encoder)
def train(diffusion_model, train_dataset, test_dataset, model_path, args, vit_fp, fp_dim):
    """
    Train the diffusion model with the given datasets and arguments.

    Parameters:
    - diffusion_model: The diffusion model to be trained.
    - train_dataset: The dataset used for training.
    - test_dataset: The dataset used for testing.
    - model_path: Path to save the trained model.
    - args: Command line arguments containing training parameters.
    - vit_fp: Whether to use precomputed feature embeddings.
    - fp_dim: Dimension of the feature embeddings.
    """
    # Extract configurations from the model and command line arguments, including training device, number of classes, total training epochs, k value in KNN, and warmup epochs.
    device = diffusion_model.device
    n_class = diffusion_model.n_class
    n_epochs = args.nepoch
    batch_size = args.batch_size
    k = args.k
    warmup_epochs = args.warmup_epochs
    lr = args.lr
    data_dir = os.path.join(os.getcwd(), args.root)
    noisy_labels = torch.tensor(train_dataset.labels).squeeze().to(device)

    # Extract features
    print('data_dir:', data_dir)
    train_embed_dir = os.path.join(data_dir, f'fp_embed_train_voc2007.npy')
    print('pre-computing fp embeddings for training data')
    train_embed= prepare_fp_x(fp_encoder, train_dataset, save_dir=train_embed_dir, device=device, fp_dim=fp_dim)
    train_embed = train_embed.to(device)

    #计算共现矩阵
    sample_variance = calculate_neighborhood_label_variance(train_embed, noisy_labels, args.k)
    clean_ind, clean_labels = select_clean_samples(noisy_labels, sample_variance, ratio=0.5)
    COM = calculate_co_occurrence_matrix(clean_labels, args.n_class)
    
    # For testing data
    print('pre-computing fp embeddings for testing data')
    test_embed_dir = os.path.join(data_dir, f'fp_embed_test_voc2007.npy')
    test_embed = prepare_fp_x(diffusion_model.fp_encoder, test_dataset, test_embed_dir, device=device, fp_dim=fp_dim)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    optimizer = optim.Adam(diffusion_model.model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
    diffusion_loss = nn.MSELoss(reduction='none')

    # Initialize EMA helper and register model parameters to smooth model parameter updates during training to improve model stability and performance.
    ema_helper = EMA(mu=0.9999)
    ema_helper.register(diffusion_model.model)

    # Train in a loop and record the highest accuracy to save the model
    max_map = 0.0
    losses = []
    print('Diffusion training start')
    for epoch in range(n_epochs):
        diffusion_model.model.train()
        total_loss = 0.0
        total_batches = 0

        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train diffusion epoch {epoch}', ncols=120) as pbar:
            for i, data_batch in pbar:
                [x_batch, y_batch, data_indices] = data_batch[:4]
                x_batch = x_batch.float().to(device)
                y_noisy = y_batch.float().to(device)

                if vit_fp:
                    # Use precomputed feature embeddings
                    fp_embd = train_embed[data_indices, :].float().to(device)
                else:
                    # Compute feature embeddings in real-time
                    fp_embd = diffusion_model.fp_encoder(x_batch.to(device))

                # Perform label sampling based on two views
                y_labels_batch, sample_weight_batch = estimate_knn_labels_matrix(fp_embd, y_noisy, train_embed,
                                                                  noisy_labels,
                                                                  k=k, n_class=n_class, weighted=True, Hard=args.hard, co_occurrence_matrix= COM)
                sample_weight_batch = sample_weight_batch.float().unsqueeze(1)
                # print(sample_weight_batch)                                                  

                # Adjust the learning rate
                adjust_learning_rate(optimizer, i / len(train_loader) + epoch, warmup_epochs=warmup_epochs, n_epochs=n_epochs, lr_input=lr)

                # Sampling t
                n = x_batch.size(0)
                t = torch.randint(low=0, high=diffusion_model.num_timesteps, size=(n // 2 + 1, )).to(device)
                t = torch.cat([t, diffusion_model.num_timesteps - 1 - t], dim=0)[:n]

                y_0_batch = y_labels_batch.float().to(device)
                
                # sampling t
                t = torch.randint(low=0, high=diffusion_model.num_timesteps, size=(n // 2 + 1,)).to(device)
                t = torch.cat([t, diffusion_model.num_timesteps - 1 - t], dim=0)[:n]

                # train with and without prior
                output, e = diffusion_model.forward_t(y_0_batch, x_batch, t, fp_embd)

                # compute loss
                mse_loss = diffusion_loss(e, output)
                # print(f"sample_weight_batch shape: {sample_weight_batch.shape}")
                # print(f"mse_loss shape: {mse_loss.shape}")
                weighted_mse_loss = sample_weight_batch * mse_loss 
                loss = torch.mean(weighted_mse_loss)
                total_loss += loss.item()
                total_batches += 1
                pbar.set_postfix({'loss': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(diffusion_model.model.parameters(), 1.0)
                optimizer.step()
                ema_helper.update(diffusion_model.model)

        average_loss = total_loss / total_batches
        losses.append(average_loss)
        # Every epoch, perform validation, if the validation accuracy of the current epoch exceeds the previous highest accuracy, evaluate the model on the test set, and save the current best model parameters.
        if epoch >= warmup_epochs:
            test_map, test_of1, test_cf1  = test(diffusion_model, test_loader, test_embed)
            logger.info(f"epoch: {epoch}, test mAP: {test_map:.2f}%, test OF1: {test_of1:.2f}%, test CF1: {test_cf1:.2f}%")
            if test_map > max_map:
                # Save diffusion model
                print('Improved!')
                states = [
                    diffusion_model.model.state_dict(),
                    diffusion_model.diffusion_encoder.state_dict(),
                    diffusion_model.fp_encoder.state_dict()
                ]
                torch.save(states, model_path)
                message = (f"Model saved and update! best mAP at Epoch {epoch}, test mAP: {test_map}, test OF1: {test_of1:.2f}%, test CF1: {test_cf1:.2f}%")
                logger.info(message)
                max_map = max(max_map, test_map)
        
    plot_and_save_losses(losses, args.log_name)

def test(diffusion_model, test_loader, test_embed):
    """
    Test the diffusion model with the given test loader and embeddings.

    Parameters:
    - diffusion_model: The diffusion model to be tested.
    - test_loader: DataLoader for the test set.
    - test_embed: Precomputed feature embeddings for the test set.

    Returns:
    - mAP: The mean average precision of the model on the test set.
    """
    if not torch.is_tensor(test_embed):
        test_embed = torch.tensor(test_embed).to(torch.float)

    with torch.no_grad():
        diffusion_model.model.eval()
        diffusion_model.diffusion_encoder.eval()
        diffusion_model.fp_encoder.eval()

        # Initialize lists to store predictions and targets
        all_preds = []
        all_targets = []

        for test_batch_idx, data_batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f'evaluating diff', ncols=100):
            [x_batch, target, indicies] = data_batch[:3]
            target = target.to(device)
            fp_embed = test_embed[indicies, :].to(device)

            # Get model predictions (probabilities for each class)
            preds = diffusion_model.reverse_ddim(x_batch, stochastic=False, fp_x=fp_embed).detach().cpu()

            # Store predictions and targets
            all_preds.append(preds)
            all_targets.append(target.cpu())

        # Concatenate all predictions and targets
        all_preds = torch.cat(all_preds, dim=0)  # Shape: [num_samples, num_classes]
        all_targets = torch.cat(all_targets, dim=0)  # Shape: [num_samples, num_classes]

        # Calculate mAP
        # mAP = compute_mAP(all_preds, all_targets)
        mAP, OF1, CF1 = compute_metrics(all_preds, all_targets) 
    return mAP*100, OF1*100, CF1*100

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123, help='Random seed for reproducibility')
    parser.add_argument('--root', type=str, default='data/2012/')
    parser.add_argument('--image_size', type=int, help='image_size', default=224)
    parser.add_argument('--n_class', type=int, default=20, help="class")
    parser.add_argument('--split_percentage', type=float, help='train and validation', default=0.9)
    parser.add_argument('--noise_rate', type=float, help='overall corruption rate, should be less than 1', default=0.3)
    parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='symmetric')
    # Training parameters
    parser.add_argument("--nepoch", default=30, help="number of training epochs", type=int)
    parser.add_argument("--batch_size", default=128, help="batch_size", type=int)
    parser.add_argument("--num_workers", default=4, help="num_workers", type=int)
    parser.add_argument("--warmup_epochs", default=0, help="warmup_epochs", type=int)
    parser.add_argument("--lr", default=5e-3, help="learning rate", type=float)
    # Diffusion model hyperparameters
    parser.add_argument("--feature_dim", default=1024, help="feature_dim", type=int)
    parser.add_argument("--k", default=50, help="k neighbors for knn or cos", type=int)
    parser.add_argument("--hard", default=False, help="soft or hard estimate", action='store_true')
    parser.add_argument("--ddim_n_step", default=10, help="number of steps in ddim", type=int)
    parser.add_argument("--diff_encoder", default='resnet50_l', help="which encoder for diffusion (linear, resnet18, 34, 50...)", type=str)
    # Large model hyperparameters
    parser.add_argument("--fp_encoder", default='ViT', help="which encoder for fp (Vit or ResNet)", type=str)
    parser.add_argument("--ViT_type", default='ViT-L/14', help="which encoder for Vit", type=str)
    # Storage path
    parser.add_argument("--gpu_devices", default=[0, 1, 2, 3], type=int, nargs='+', help="")
    parser.add_argument("--device", default='cuda:0', help="which cuda to use", type=str)
    parser.add_argument("--log_path", default='./logs', help="input your logs path", type=str)
    parser.add_argument("--log_name", default='Voc2012', help="create your logs name", type=str)
    args = parser.parse_args()
    logger = setup_logger(args, log_dir = args.log_path)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.device is None:
        gpu_devices = ','.join([str(id) for id in args.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device


    # Prepare dataset directories
    print('data_dir', args.root)

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
            MultiScaleCrop(args.image_size, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    val_transform= transforms.Compose([
            Warp(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # Load datasets
    train_dataset = Voc2007Classification(args.root,
                                set_name='train', 
                                img_size= args.image_size,
                                noise_type=args.noise_type, 
                                noise_rate=args.noise_rate, 
                                transform=train_transform,
                                split_per=args.split_percentage,
                                random_seed=args.seed)

    val_dataset = Voc2007Classification(args.root,
                                set_name='val', 
                                img_size= args.image_size,
                                noise_type=args.noise_type, 
                                noise_rate=args.noise_rate, 
                                transform=val_transform,
                                split_per=args.split_percentage,
                                random_seed=args.seed)

    test_dataset = Voc2007Classification(args.root, 
                            set_name='test',
                            img_size= args.image_size,
                            transform=val_transform)


    # Load fp feature extractor
    fp_encoder = vit_img_wrap(args.ViT_type, args.device, center=mean, std=std)
    fp_dim = fp_encoder.dim

    # Initialize the diffusion model
    model_path = './model/CAD_Voc2007.pt'
    diffusion_model = Diffusion(fp_encoder, num_timesteps=1000, n_class=args.n_class, fp_dim=fp_dim, device=device, feature_dim=args.feature_dim, encoder_type=args.diff_encoder, ddim_num_steps=args.ddim_n_step, beta_schedule='cosine')
    # Load model weights
    # state_dict = torch.load(model_path, map_location=torch.device(device))
    # diffusion_model.load_diffusion_net(state_dict)

    # DataParallel wrapper
    if args.device is None:
        print('using DataParallel')
        diffusion_model.model = nn.DataParallel(diffusion_model.model).to(device)
        diffusion_model.diffusion_encoder = nn.DataParallel(diffusion_model.diffusion_encoder).to(device)
        diffusion_model.fp_encoder = nn.DataParallel(fp_encoder).to(device)
    else:
        print('using single gpu: ', device)
        diffusion_model.to(device)

    # Train the diffusion model
    train(diffusion_model, train_dataset, test_dataset, model_path, args, vit_fp=True, fp_dim=fp_dim)





