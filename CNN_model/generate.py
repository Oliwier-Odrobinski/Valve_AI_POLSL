import torch
import os
import re
import pandas as pd
from model import Gen  # Import klasy modelu
import torchvision.utils as vutils

CHECKPOINT_DIR = 'checkpoints'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_MANUAL_LIST = True 

if USE_MANUAL_LIST:
    OUTPUT_DIR = 'generated_results_manual'
    input_list = [
    (0, 0.45, 10),
    (0, 0.53, 10),
    (0, 1.1, 50),
    (0, 2.6, 50),
    (0, 3.5, 50),
    (0, 3.4, 80),
    (0, 5.3, 80),
    (0, 6.45, 80)
    ]
else:
    OUTPUT_DIR = 'generated_results'
    CSV_PATH = 'interpolated_dane_for_gen.csv'
    
    _df = pd.read_csv(CSV_PATH, sep=';')
    _df.columns = _df.columns.str.strip()
    
    input_list = list(zip(_df['t'], _df['przeplyw'].astype(float), _df['otwarcie'].astype(float)))

def generate_from_checkpoints():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')]
    checkpoints = [f for f in checkpoints if any((int(n) >= 6000 and int(n)%500==0)for n in re.findall(r'\d+', f))]
    checkpoints.sort() 

    if not checkpoints:
        print(f"Nie znaleziono plików .pth w folderze {CHECKPOINT_DIR}")
        return

    model = Gen().to(DEVICE)
    model.eval() 

    print(f"Rozpoczynam generowanie dla {len(checkpoints)} modeli...")

    with torch.no_grad(): 
        for ckpt_name in checkpoints:
            ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
            print(f"Przetwarzam: {ckpt_name}")

            checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            max_str, max_otw = checkpoint['max_values']


            model_output_dir = os.path.join(OUTPUT_DIR, ckpt_name.replace('.pth', ''))
            os.makedirs(model_output_dir, exist_ok=True)

            for i, (val_t, val_str, val_otw) in enumerate(input_list):
                norm_str = val_str / max_str
                norm_otw = val_otw / max_otw
                

                input_tensor = torch.tensor([[norm_str, norm_otw]], dtype=torch.float32).to(DEVICE)

                generated_img = model(input_tensor)

                save_path = os.path.join(model_output_dir, f'{val_t}_{val_otw}_{val_str}.png')
                vutils.save_image(generated_img, save_path, normalize=True)



if __name__ == '__main__':
    generate_from_checkpoints()