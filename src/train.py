import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import load_config
from data import TreeDataset
from model import UNet
from utils import count_mae, count_rmse

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, dens, _ in tqdm(loader, desc='  Train', leave=False):
        imgs, dens = imgs.to(device), dens.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), dens)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = mae = rmse = 0.0
    with torch.no_grad():
        for imgs, dens, _ in tqdm(loader, desc='  Val  ', leave=False):
            imgs, dens = imgs.to(device), dens.to(device)
            out  = model(imgs)
            loss = criterion(out, dens)
            total_loss += loss.item() * imgs.size(0)
            for i in range(imgs.size(0)):
                pred_np = out[i, 0].cpu().numpy()
                gt_np   = dens[i, 0].cpu().numpy()
                mae  += count_mae(pred_np, gt_np)
                rmse += count_rmse(pred_np, gt_np)
    n = len(loader.dataset)
    return total_loss / n, mae / n, rmse / n

def main():
    config = load_config("../config.yml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths in config are relative to config.yml (which is now ../ relative to src)
    # We'll assume the relative paths in config.yml (e.g., '../dataset/...') 
    # are evaluated from the root folder if we run from the root folder.
    # To make it robust regardless of where we run, we resolve relative to the project root.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def resolve_path(p):
        return os.path.normpath(os.path.join(base_dir, p.lstrip("./\\")))

    train_ds = TreeDataset(
        resolve_path(config['data']['train_img']), 
        resolve_path(config['data']['train_mask']),
        gaussian_sigma=config['processing']['gaussian_sigma']
    )
    test_ds  = TreeDataset(
        resolve_path(config['data']['test_img']),  
        resolve_path(config['data']['test_mask']),
        gaussian_sigma=config['processing']['gaussian_sigma']
    )

    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(in_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    criterion = nn.MSELoss()

    best_mae = float('inf')
    epochs = config['training']['epochs']
    model_save_path = resolve_path(config['training']['model_save_path'])

    for epoch in range(epochs):
        lr_before = optimizer.param_groups[0]['lr']
        print(f'\nEpoch [{epoch+1:>2}/{epochs}]  |  LR: {lr_before:.2e}')

        t_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_mae, v_rmse = eval_epoch(model, test_loader, criterion, device)

        scheduler.step(v_mae)
        lr_after = optimizer.param_groups[0]['lr']
        if lr_after != lr_before:
            print(f'LR reduced: {lr_before:.2e} -> {lr_after:.2e}')

        print(f'Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val MAE: {v_mae:.2f} | Val RMSE: {v_rmse:.2f}')

        if v_mae < best_mae:
            best_mae = v_mae
            
            # create dir if doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
            
            torch.save(model.state_dict(), model_save_path)
            print(f'Best model saved  (MAE={best_mae:.2f})')

    print(f'\nTraining complete. Best Val MAE: {best_mae:.2f}')

if __name__ == '__main__':
    main()
