import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import load_data, make_dataloaders
from unet import unet_model, get_optimizer, train_step, val_step

# config
DATA_DIR      = './dataset/'
EPOCHS        = 60          # more epochs, early stopping will cut short
BATCH_SIZE    = 16
PATCHES_TRAIN = 2000
PATCHES_VAL   = 500
PATCH_SZ      = 160
LR            = 1e-3
WEIGHT_DECAY  = 1e-4       # L2 regularisation
DROPRATE      = 0.4        # higher dropout for small dataset
N_FILTERS     = 16         # smaller model (was 32) → less overfitting
PATIENCE      = 10         # early stopping patience
SAVE_PATH     = './best_model.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using: {device}')

X_TR, Y_TR, X_TE, Y_TE = load_data(DATA_DIR)

train_dl, val_dl = make_dataloaders(
    X_TR, Y_TR, X_TE, Y_TE,
    n_patches_train=PATCHES_TRAIN,
    n_patches_val=PATCHES_VAL,
    sz=PATCH_SZ,
    batch_size=BATCH_SIZE,
)

model     = unet_model(n_classes=1, n_channels=4, im_sz=PATCH_SZ,
                       n_filters_start=N_FILTERS, droprate=DROPRATE).to(device)
optimiser = get_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=5)

best_val_loss  = float('inf')
no_improve     = 0
current_lr     = LR

for epoch in range(1, EPOCHS + 1):
    train_dl.dataset._resample()
    val_dl.dataset._resample()

    # train
    loss, acc, prec, rec = 0, 0, 0, 0
    for x, y in train_dl:
        m     = train_step(model, optimiser, x, y, device)
        loss += m['loss'];  acc  += m['accuracy']
        prec += m['precision']; rec += m['recall']
    n = len(train_dl)
    loss /= n;  acc /= n;  prec /= n;  rec /= n

    # validate
    vloss, vacc, vprec, vrec = 0, 0, 0, 0
    for x, y in val_dl:
        m      = val_step(model, x, y, device)
        vloss += m['val_loss'];  vacc  += m['val_accuracy']
        vprec += m['val_precision']; vrec += m['val_recall']
    n = len(val_dl)
    vloss /= n;  vacc /= n;  vprec /= n;  vrec /= n

    scheduler.step(vloss)
    new_lr = optimiser.param_groups[0]['lr']
    if new_lr != current_lr:
        print(f'  LR reduced to {new_lr:.2e}')
    current_lr = new_lr

    print(
        f"[{epoch:03d}/{EPOCHS}] lr {current_lr:.2e} | "
        f"loss {loss:.4f}  acc {acc:.4f}  prec {prec:.4f}  rec {rec:.4f} | "
        f"val_loss {vloss:.4f}  val_acc {vacc:.4f}  val_prec {vprec:.4f}  val_rec {vrec:.4f}"
    )

    if vloss < best_val_loss:
        best_val_loss = vloss
        no_improve    = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print(f'  --> best model saved (val_loss={best_val_loss:.4f})')
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f'  Early stopping: no improvement for {PATIENCE} epochs.')
            break

print(f'\nDone. Best val_loss: {best_val_loss:.4f}  ->  {SAVE_PATH}')
