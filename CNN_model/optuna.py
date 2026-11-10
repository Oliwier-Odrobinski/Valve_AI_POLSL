import os
import gc
import json
import random
from collections import Counter

import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from dataset import CustomImageDataset
from model import Gen

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
CSV_PATH = 'data/data.csv'
IMG_DIR = 'data/images'
OUTPUT_DIR = 'optuna_results'
EPOCHS_PER_TRIAL = 200
N_TRIALS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_datasets():
    dataset = CustomImageDataset(
        csv_file=CSV_PATH,
        img_dir=IMG_DIR
    )

    total_size = len(dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(SEED)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    return dataset, train_dataset, val_dataset


def prepare_dataloaders(dataset, train_dataset, val_dataset, batch_size):
    train_indices = train_dataset.indices

    train_openings = dataset.data_frame.iloc[
        train_indices
    ]['otwarcie'].to_numpy()

    _, inverse, counts = np.unique(
        train_openings,
        return_inverse=True,
        return_counts=True
    )

    sample_weights = 1.0 / counts[inverse]

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(SEED)
    )

    pin_memory = DEVICE.type == 'cuda'

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


def objective(trial, dataset, train_dataset, val_dataset):
    set_seed(SEED)

    learning_rate = trial.suggest_float(
        'learning_rate',
        0.001,
        0.004
    )

    batch_size = trial.suggest_categorical(
        'batch_size',
        [2, 4]
    )

    norm_type = trial.suggest_categorical(
        'norm_type',
        ['batch', 'instance']
    )

    embedding_activation = trial.suggest_categorical(
        'embedding_activation',
        ['relu', 'leaky_relu']
    )

    conv_activation = trial.suggest_categorical(
        'conv_activation',
        ['relu', 'leaky_relu']
    )

    train_loader, val_loader = prepare_dataloaders(
        dataset,
        train_dataset,
        val_dataset,
        batch_size
    )

    model = Gen(
        input_dim=2,
        norm_type=norm_type,
        embedding_activation=embedding_activation,
        conv_activation=conv_activation
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    final_val_loss = None

    try:
        for epoch in range(EPOCHS_PER_TRIAL):
            model.train()

            for params, real_images in train_loader:
                params = params.to(DEVICE)
                real_images = real_images.to(DEVICE)

                optimizer.zero_grad()
                generated_images = model(params)
                loss = criterion(generated_images, real_images)
                loss.backward()
                optimizer.step()

            model.eval()
            running_val_loss = 0.0
            val_samples = 0

            with torch.no_grad():
                for params, real_images in val_loader:
                    params = params.to(DEVICE)
                    real_images = real_images.to(DEVICE)

                    generated_images = model(params)
                    val_loss = criterion(generated_images, real_images)

                    running_val_loss += val_loss.item() * params.size(0)
                    val_samples += params.size(0)

            final_val_loss = running_val_loss / val_samples

            trial.report(final_val_loss, step=epoch + 1)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return final_val_loss

    finally:
        del model
        del optimizer
        del train_loader
        del val_loader

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_optimization():
    dataset, train_dataset, val_dataset = prepare_datasets()

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=20
    )

    sampler = optuna.samplers.TPESampler(seed=SEED)

    study = optuna.create_study(
        study_name='valve_ai_optimization',
        storage='sqlite:///optuna_valve.db',
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )

    study.optimize(
        lambda trial: objective(
            trial,
            dataset,
            train_dataset,
            val_dataset
        ),
        n_trials=N_TRIALS,
        gc_after_trial=True,
        show_progress_bar=False
    )

    trials_df = study.trials_dataframe()
    trials_df.to_csv(
        os.path.join(OUTPUT_DIR, 'all_trials.csv'),
        index=False
    )

    completed_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
    ]

    if not completed_trials:
        return

    best_params = study.best_params

    with open(os.path.join(OUTPUT_DIR, 'best_params.json'), 'w') as f:
        json.dump(
            {
                'best_validation_mse': study.best_value,
                'parameters': best_params
            },
            f,
            indent=4
        )

    top5 = sorted(
        completed_trials,
        key=lambda trial: trial.value
    )[:5]

    top5_results = []

    for rank, trial in enumerate(top5, start=1):
        result = {
            'rank': rank,
            'validation_mse': trial.value,
            **trial.params
        }
        top5_results.append(result)

    pd.DataFrame(top5_results).to_csv(
        os.path.join(OUTPUT_DIR, 'top5_trials.csv'),
        index=False
    )

    top_params = [trial.params for trial in top5]

    final_params = {
        'learning_rate': float(np.median([
            p['learning_rate'] for p in top_params
        ])),
        'batch_size': Counter(
            p['batch_size'] for p in top_params
        ).most_common(1)[0][0],
        'norm_type': Counter(
            p['norm_type'] for p in top_params
        ).most_common(1)[0][0],
        'embedding_activation': Counter(
            p['embedding_activation'] for p in top_params
        ).most_common(1)[0][0],
        'conv_activation': Counter(
            p['conv_activation'] for p in top_params
        ).most_common(1)[0][0]
    }

    with open(
        os.path.join(OUTPUT_DIR, 'top5_representative_params.json'),
        'w'
    ) as f:
        json.dump(final_params, f, indent=4)


if __name__ == '__main__':
    run_optimization()