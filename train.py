
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


args = {
    'model': 'gunet_t',
    'num_workers': 0,  # Set to 0 to avoid multiprocessing issues in Kaggle
    'use_mp': True,  # Use mixed precision for faster training
    'use_ddp': False,  # Disable DDP in Kaggle
    'save_dir': '/kaggle/working/saved_models/',
    'data_dir': '/kaggle/input/',
    'log_dir': '/kaggle/working/logs/',
    'train_set': 'haze4k-t/Haze4K-T',
    'val_set': 'haze4k-v/Haze4K-V',
    'exp': 'reside-in'
}

# Load configs
with open(os.path.join('/kaggle/working/configs', args['exp'], 'base.json'), 'r') as f:
    b_setup = json.load(f)

variant = args['model'].split('_')[-1]
config_name = 'model_' + variant + '.json'
with open(os.path.join('/kaggle/working/configs', args['exp'], config_name), 'r') as f:
    m_setup = json.load(f)

def train(train_loader, network, criterion, optimizer, scaler, frozen_bn=False):
    losses = AverageMeter()
    torch.cuda.empty_cache()
    network.eval() if frozen_bn else network.train()

    for batch in train_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with autocast('cuda', enabled=args['use_mp']):
            output = network(source_img)
            loss = criterion(output, target_img)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item())

    return losses.avg

def valid(val_loader, network):
    PSNR = AverageMeter()
    torch.cuda.empty_cache()
    network.eval()

    for batch in val_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with torch.no_grad():
            H, W = source_img.shape[2:]
            source_img = pad_img(source_img, network.patch_size)
            output = network(source_img).clamp_(-1, 1)
            output = output[:, :, :H, :W]

        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), source_img.size(0))

    return PSNR.avg

def main():
    network = gunet_t()
    network.cuda()

    criterion = nn.L1Loss()

    optimizer = torch.optim.AdamW(network.parameters(), lr=m_setup['lr'], weight_decay=b_setup['weight_decay'])
    lr_scheduler = CosineScheduler(optimizer, param_name='lr', t_max=b_setup['epochs'], value_min=m_setup['lr'] * 1e-2,
                                   warmup_t=b_setup['warmup_epochs'], const_t=b_setup['const_epochs'])
    wd_scheduler = CosineScheduler(optimizer, param_name='weight_decay', t_max=b_setup['epochs'])
    scaler = GradScaler('cuda')

    save_dir = os.path.join(args['save_dir'], args['exp'])
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(os.path.join(save_dir, args['model'] + '.pth')):
        best_psnr = 0
        cur_epoch = 0
    else:
        print('==> Loaded existing trained model.')
        model_info = torch.load(os.path.join(save_dir, args['model'] + '.pth'), map_location='cpu')
        network.load_state_dict(model_info['state_dict'])
        optimizer.load_state_dict(model_info['optimizer'])
        lr_scheduler.load_state_dict(model_info['lr_scheduler'])
        wd_scheduler.load_state_dict(model_info['wd_scheduler'])
        scaler.load_state_dict(model_info['scaler'])
        cur_epoch = model_info['cur_epoch']
        best_psnr = model_info['best_psnr']

    train_dataset = PairLoader(os.path.join(args['data_dir'], args['train_set']), 'train',
                               b_setup['t_patch_size'],
                               b_setup['edge_decay'],
                               b_setup['data_augment'],
                               b_setup['cache_memory'])
    train_loader = DataLoader(train_dataset,
                              batch_size=m_setup['batch_size'],
                              sampler=RandomSampler(train_dataset, num_samples=b_setup['num_iter']),
                              num_workers=args['num_workers'],  # Now 0
                              pin_memory=True,
                              drop_last=True)

    val_dataset = PairLoader(os.path.join(args['data_dir'], args['val_set']), b_setup['valid_mode'],
                             b_setup['v_patch_size'])
    val_loader = DataLoader(val_dataset,
                            batch_size=max(int(m_setup['batch_size'] * b_setup['v_batch_ratio']), 1),
                            num_workers=args['num_workers'],  # Now 0
                            pin_memory=True)

    print('==> Start training, current model name: ' + args['model'])
    writer = SummaryWriter(log_dir=os.path.join(args['log_dir'], args['exp'], args['model']))

    for epoch in tqdm(range(cur_epoch, b_setup['epochs'] + 1)):
        frozen_bn = epoch > (b_setup['epochs'] - b_setup['frozen_epochs'])

        loss = train(train_loader, network, criterion, optimizer, scaler, frozen_bn)
        lr_scheduler.step(epoch + 1)
        wd_scheduler.step(epoch + 1)

        writer.add_scalar('train_loss', loss, epoch)

        if epoch % b_setup['eval_freq'] == 0:
            avg_psnr = valid(val_loader, network)

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save({'cur_epoch': epoch + 1,
                            'best_psnr': best_psnr,
                            'state_dict': network.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'wd_scheduler': wd_scheduler.state_dict(),
                            'scaler': scaler.state_dict()},
                           os.path.join(save_dir, args['model'] + '.pth'))

            writer.add_scalar('valid_psnr', avg_psnr, epoch)
            writer.add_scalar('best_psnr', best_psnr, epoch)

if __name__ == '__main__':
    main()
