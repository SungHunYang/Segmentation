import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import tqdm
import logging
from datetime import datetime

from utils.custom_dataset import MyDataset
from utils.load_model import get_model
from utils.utils import iou_calculater
from utils.dice_loss import DiceLoss

device = "cuda" if torch.cuda.is_available() else "cpu"
#####################################################################################
# 0: 'deeplabv3plus_resnet50'
# 1: 'deeplabv3plus_mobilenet'
# 2: 'unet'
# 3: 'unext'

pick_idx = 3
net, net_name = get_model(pick_idx, device)

# PARMS
batch_size = 64
max_epoch = 100
learning_rate = 0.001
num_workers = 0
pin_memory = True

# SET OPTIMIZER & LOSS FUNCTION
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
criterion = DiceLoss().to(device)
# criterion = nn.BCEWithLogitsLoss().to(device)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: min(2. * (max_epoch - epoch) / max_epoch, 1.))
#####################################################################################
# SET TIME AND DIRECTORY
nows = datetime.now().strftime('%Y%m%d_%H%M')
path = f'./Checkpoints/{nows}'

os.makedirs(path, exist_ok=True)
os.makedirs(f'{path}/model', exist_ok=True)
os.makedirs(f'{path}/state', exist_ok=True)

# SETUP LOGGER
logging.basicConfig(filename=f'{path}/train_log.log',
                    format='{asctime}.{msecs:03.0f} | {levelname} | {message}',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    style='{',
                    level=logging.DEBUG)

log_console = logging.StreamHandler()
log_console.setLevel(logging.INFO)
log_console.setFormatter(logging.Formatter(fmt='{asctime} | {message}', datefmt='%H:%M:%S', style='{'))
logging.getLogger('').addHandler(log_console)
logging.getLogger('PIL').setLevel(logging.WARNING)

# DATA LOADER (train, valid only)
train_path = 'C:/dataset/segmentation_outline_split/train'
valid_path = 'C:/dataset/segmentation_outline_split/valid'
train_imgs = MyDataset(train_path)
valid_imgs = MyDataset(valid_path)

train_dataloader = DataLoader(train_imgs, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
valid_dataloader = DataLoader(valid_imgs, batch_size=batch_size, shuffle=False, num_workers=num_workers)
# LOG INFO
loss_name = criterion.__class__.__name__
opt_name = optimizer.__class__.__name__

# TRAINING
start_epoch = 1

min_val_loss = 1
min_val_epoch = 0

start_epoch_time = datetime.now().timestamp()

logging.info('model %s | input %s | lossf %s | opt %s | lr %.3f | batch %d | epochs %d',
             net_name, 'outline', loss_name, opt_name, learning_rate, batch_size, max_epoch)

for epoch in range(start_epoch, max_epoch + 1):
    net.train()

    train_loss_sum = 0
    val_loss_sum = 0
    val_iou_sum = 0

    for i, data_ in enumerate(tqdm.tqdm(train_dataloader)):
        inputs, labels = data_
        inputs, labels = inputs.to(device), labels.to(device)

        preds = net(inputs)

        optimizer.zero_grad()

        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss

    scheduler.step()

    with torch.no_grad():
        net.eval()
        for data_ in valid_dataloader:
            inputs, labels = data_
            inputs, labels = inputs.to(device), labels.to(device)

            preds = net(inputs)

            loss = criterion(preds, labels)
            iou = iou_calculater(preds, labels)

            val_loss_sum += loss
            val_iou_sum += iou

    train_loss = train_loss_sum / len(train_dataloader)
    val_loss = val_loss_sum / len(valid_dataloader)
    val_iou = val_iou_sum / len(valid_dataloader)

    logging.info('Epoch %d | train_loss %.4f | val_loss %.4f | val_iou %.4f',
                 epoch, train_loss, val_loss, val_iou)

    # ESTIMATE TIME
    estimated_seconds_left = (datetime.now().timestamp() - start_epoch_time) * \
                             (max_epoch - start_epoch + 1) / (epoch - start_epoch + 1)
    estimated_finish_time = datetime.fromtimestamp(start_epoch_time + estimated_seconds_left)
    estimated_finish_time = estimated_finish_time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'et: {estimated_finish_time}', end=',')

    # SAVE STATE (for resume)
    if val_loss < min_val_loss:
        min_val_epoch = epoch
        min_val_loss = val_loss
        torch.save(net.state_dict(), f'{path}/model/epoch_{"%03d" % epoch}.pt')

    print(f' best epoch: {min_val_epoch}\n{"=" * 110}')

    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch,
            'net_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, f'{path}/state/state_latest.pt.tar')
