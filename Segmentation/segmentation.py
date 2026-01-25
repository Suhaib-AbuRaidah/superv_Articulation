import os
import torch
import datetime
import sys
import numpy as np
sys.path.append('/home/suhaib/superv_Articulation')
import tqdm
from torch.utils.tensorboard import SummaryWriter

from utilis.dataset2 import Segmentation_Dataset
from torch.utils.data import DataLoader 
from PointNet2.models.pointnet2_part_seg_msg import get_model, get_loss
from PointNet2 import provider


# -----------------------
# Config
# -----------------------
BATCH_SIZE = 8
NUM_EPOCHS = 20
NUM_CLASSES = 8
LEARNING_RATE = 1e-3

WEIGHT_DECAY = 0.0
LR_DECAY_STEP = 4
LR_DECAY_GAMMA = 0.8

# -----------------------
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Datasets / Loaders
# -----------------------
train_dataset = Segmentation_Dataset(
    "../Ditto/Articulated_object_simulation-main/data/Shape2Motion_gcn/robotic_arm/train/scenes/*.npz",
    device
)
val_dataset = Segmentation_Dataset(
    "../Ditto/Articulated_object_simulation-main/data/Shape2Motion_gcn/robotic_arm/val/scenes/*.npz",
    device
)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# -----------------------
# Model / Optim / Loss
# -----------------------
model = get_model(num_classes=NUM_CLASSES, normal_channel=False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)

criterion = get_loss().to(device)

# -----------------------
# Logging / Checkpoints
# -----------------------
start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
checkpoint_path = f"./Segmentation/pre_trained_models_seg/{start_time}_robotic_arm"
os.makedirs(checkpoint_path, exist_ok=True)

writer = SummaryWriter(f"runs/{start_time}_robotic_arm")
print(f"tensorboard --logdir runs/{start_time}_robotic_arm")
print(f"Checkpoints will be saved to: {checkpoint_path}")

best_loss_train = float("inf")
best_loss_val   = float("inf")

# -----------------------
# Training Loop
# -----------------------
for epoch in range(NUM_EPOCHS):
    current_lr = scheduler.get_last_lr()[0]
    print(f"\n=== Epoch {epoch+1} | LR: {current_lr:.6f} ===")

    # -------- Train --------
    model.train()
    epoch_train_loss = 0.0

    pbar_train = tqdm.tqdm(
        train_dataloader,
        desc=f"Train Ep {epoch+1}",
        leave=True,
        dynamic_ncols=True
    )

    for step, data in enumerate(pbar_train, start=1):
        pc_start, mask_start_list, pc_end, mask_end_list = data

        pc_start  = pc_start.to(device)          # (B, N, 3)
        seg_labels = mask_start_list.to(device)  # (B, N)

        optimizer.zero_grad()

        preds, _ = model(pc_start.transpose(2, 1), None)
        # preds: (B, N, C)

        preds = preds.reshape(-1, NUM_CLASSES)
        seg_labels = seg_labels.view(-1)

        loss_val = criterion(preds, seg_labels, None)
        loss_val.backward()
        optimizer.step()

        epoch_train_loss += loss_val.item()
        avg_train_loss = epoch_train_loss / step

        pbar_train.set_postfix({
            "Loss": f"{loss_val.item():.4f}",
            "Avg": f"{avg_train_loss:.4f}"
        })

    avg_train_loss = epoch_train_loss / len(train_dataloader)
    writer.add_scalar("Loss/train", avg_train_loss, epoch)

    # -------- Validation --------
    model.eval()
    epoch_val_loss = 0.0

    pbar_val = tqdm.tqdm(
        val_dataloader,
        desc=f"Val Ep {epoch+1}",
        leave=True,
        dynamic_ncols=True
    )

    with torch.no_grad():
        for val_step, val_data in enumerate(pbar_val, start=1):
            pc_start, mask_start_list, pc_end, mask_end_list = val_data

            pc_start  = pc_start.to(device)
            seg_labels = mask_start_list.to(device)

            preds, _ = model(pc_start.transpose(2, 1), None)

            preds = preds.reshape(-1, NUM_CLASSES)
            seg_labels = seg_labels.view(-1)

            val_loss = criterion(preds, seg_labels, None)

            epoch_val_loss += val_loss.item()
            avg_val_loss = epoch_val_loss / val_step

            pbar_val.set_postfix({
                "Loss": f"{val_loss.item():.4f}",
                "Avg": f"{avg_val_loss:.4f}"
            })

    avg_val_loss = epoch_val_loss / len(val_dataloader)
    writer.add_scalar("Loss/val", avg_val_loss, epoch)

    scheduler.step()

    # -------- Epoch Summary --------
    print(
        f"Summary Ep {epoch+1}:\n"
        f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
    )

    # -------- Checkpoints --------
    if avg_train_loss < best_loss_train:
        best_loss_train = avg_train_loss
        torch.save(
            model.state_dict(),
            os.path.join(checkpoint_path, "chkpt_best_model_train.pth")
        )
        print(f"New best TRAIN model saved (loss={best_loss_train:.4f})")

    if avg_val_loss < best_loss_val:
        best_loss_val = avg_val_loss
        torch.save(
            model.state_dict(),
            os.path.join(checkpoint_path, "chkpt_best_model_val.pth")
        )
        print(f"New best VAL model saved (loss={best_loss_val:.4f})")

    if (epoch + 1) % 50 == 0:
        torch.save(
            model.state_dict(),
            os.path.join(checkpoint_path, f"chkpt_{epoch+1}.pth")
        )
