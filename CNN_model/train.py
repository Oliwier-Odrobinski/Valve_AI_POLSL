import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
import numpy as np
from dataset import CustomImageDataset
from model import Gen
import matplotlib.pyplot as plt

BATCH_SIZE = 2 
LEARNING_RATE = 0.0035
START_EPOCH = 0 # 
EPOCHS = 10000               
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = 'data/data.csv'
IMG_DIR = 'data/images'
CHECKPOINT_DIR = 'checkpoints'

RESUME_CHECKPOINT = os.path.join(CHECKPOINT_DIR, f'gen_model_epoch_{START_EPOCH}.pth') 
LOG_FILE = 'loss_log.csv'

def train():
    global START_EPOCH 
    print(f"Device: {DEVICE}")
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    dataset = CustomImageDataset(csv_file=CSV_PATH, img_dir=IMG_DIR)


    total_size = len(dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


    otwarcia = dataset.data_frame['otwarcie'].values
    train_indices = train_dataset.indices
    train_otwarcia = otwarcia[train_indices]

    unikalne_otwarcia, zliczenia = np.unique(train_otwarcia, return_counts=True)
    wagi_klas = 1.0 / zliczenia
    waga_dla_otwarcia = dict(zip(unikalne_otwarcia, wagi_klas))

    wagi_probek_train = [waga_dla_otwarcia[o] for o in train_otwarcia]
    wagi_probek_tensor = torch.DoubleTensor(wagi_probek_train)

    sampler = WeightedRandomSampler(
        weights=wagi_probek_tensor, 
        num_samples=len(wagi_probek_tensor), 
        replacement=True
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    max_str, max_otw = dataset.get_max_values()

    model = Gen().to(DEVICE)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if os.path.isfile(RESUME_CHECKPOINT):
        print(f"Znaleziono checkpoint: {RESUME_CHECKPOINT}. Wczytywanie wag...")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        max_str, max_otw = checkpoint['max_values']
        print(f"Pomyślnie wczytano wagi z epoki {START_EPOCH}.")
    else:
        print(f"UWAGA: Nie znaleziono {RESUME_CHECKPOINT}. Trening rozpocznie się od zera (losowe wagi).")
        START_EPOCH = 0

    if not os.path.isfile(LOG_FILE) or START_EPOCH == 0:
        with open(LOG_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss'])

    train_loss_history = []
    val_loss_history = []

    for epoch in range(START_EPOCH, EPOCHS):
        
        model.train() 
        running_train_loss = 0.0
        
        for i, (params, real_images) in enumerate(train_dataloader):
            params = params.to(DEVICE)
            real_images = real_images.to(DEVICE)

            optimizer.zero_grad()
            generated_images = model(params)

            loss = criterion(generated_images, real_images)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_dataloader)
        train_loss_history.append(epoch_train_loss)
        
        model.eval() 
        running_val_loss = 0.0
        
        with torch.no_grad():
            for params, real_images in val_dataloader:
                params = params.to(DEVICE)
                real_images = real_images.to(DEVICE)
                
                generated_images = model(params)
                val_loss = criterion(generated_images, real_images)
                
                running_val_loss += val_loss.item()
                
        epoch_val_loss = running_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
        val_loss_history.append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        with open(LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, epoch_train_loss, epoch_val_loss])
            
        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'gen_model_epoch_{epoch+1}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'max_values': (max_str, max_otw),
                'optimizer_state_dict': optimizer.state_dict() 
            }, checkpoint_path)

    torch.save({
        'model_state_dict': model.state_dict(),
        'max_values': (max_str, max_otw)
    }, 'gen_model.pth')

    plt.figure(figsize=(10, 6))
    plt.plot(range(START_EPOCH + 1, EPOCHS + 1), train_loss_history, label='Training Loss', color='blue')
    plt.plot(range(START_EPOCH + 1, EPOCHS + 1), val_loss_history, label='Validation Loss', color='red', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'learning_curve_{EPOCHS}_epoch.png', dpi=300)

if __name__ == '__main__':
    train()