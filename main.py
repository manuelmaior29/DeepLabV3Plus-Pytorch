from tqdm import tqdm
import json
import network
import utils
import os
import random
import argparse
import numpy as np
import torchvision

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, HybridDataset
from datasets import visualize_class_distribution
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes', 'hybrid_dataset'], help='Name of dataset')
    parser.add_argument("--max_train_examples", type=int, default=100,
                        help="max number of train examples from the dataset")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")
    parser.add_argument("--mixed_hybrid", action='store_true', default=False,
                        help='If hybrid dataset used, mix CS with Carla data')
    parser.add_argument("--mixed_hybrid_synthetic_examples", type=int, default=100,
                        help="max number of SYNTHETIC train examples from the dataset (if mixed_hybrid=true)")
    parser.add_argument("--mixed_hybrid_real_examples", type=int, default=100,
                        help="max number of REAL train examples from the dataset (if mixed_hybrid=true)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--pretrained_backbone", action='store_true', default=False, 
                        help='use pretrained backbone')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument('--freeze_layers', help='delimited layer list input', type=str, default='')
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=11,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')
    
    # Hybrid Dataset Options
    parser.add_argument("--data_source", type=str, default='real',
                        choices=['real', 'synthetic'], help='data source for Hybrid Dataset')
    
    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='8067',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)

    if opts.dataset == 'hybrid_dataset':

        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ]) 

        val_dst = HybridDataset(root_path=opts.data_root + '\\real\\val',
                        input_dir='rgb',
                        target_dir='semantic_segmentation_mapped',
                        transform=val_transform,
                        type='real')

        test_dst = HybridDataset(root_path=opts.data_root + '\\real\\train',
                                input_dir='rgb',
                                target_dir='semantic_segmentation_mapped',
                                transform=val_transform,
                                type='real')

        if opts.mixed_hybrid:

            train_dst_synthetic = HybridDataset(root_path=opts.data_root + '\\synthetic\\train',
                                    input_dir='rgb',
                                    target_dir='semantic_segmentation_mapped',
                                    transform=train_transform,
                                    type='synthetic')
            
            train_dst_real = HybridDataset(root_path=opts.data_root + '\\real\\train',
                                    input_dir='rgb',
                                    target_dir='semantic_segmentation_mapped',
                                    transform=train_transform,
                                    type='real')
 
            # Creation of REAL and SYNTHETIC, entire-dataset, indices
            train_dst_real_indices = list(np.arange(0, len(train_dst_real), 1))
            train_dst_synthetic_indices = list(np.arange(0, len(train_dst_synthetic), 1))
            test_dst_indices = list(np.arange(0, len(test_dst), 1))

            # Shuffling of REAL and SYNTHETIC, entire-dataset, indices
            temp_indices = list(zip(train_dst_real_indices, test_dst_indices))
            random.Random(opts.random_seed).shuffle(temp_indices)
            train_dst_real_indices, test_dst_indices = zip(*temp_indices)
            train_dst_real_indices, test_dst_indices = list(train_dst_real_indices), list(test_dst_indices)
            random.Random(opts.random_seed).shuffle(train_dst_synthetic_indices)

            # Slicing of REAL and SYNTHETIC, entire-dataset, indices (for REAL train, last 500 are for Testing)
            train_dst_real_indices = train_dst_real_indices[0:opts.mixed_hybrid_real_examples]
            train_dst_synthetic_indices = train_dst_synthetic_indices[0:opts.mixed_hybrid_synthetic_examples]
            test_dst_indices = test_dst_indices[len(test_dst)-500:len(test_dst)]

            # Initializing subsets based on previously computed indices
            train_dst_real = data.Subset(dataset=train_dst_real, indices=train_dst_real_indices) # Filter indices
            train_dst_synthetic = data.Subset(dataset=train_dst_synthetic, indices=train_dst_synthetic_indices) # Filter indices
            test_dst = data.Subset(dataset=test_dst, indices=test_dst_indices) # Filter indices

            # Concatenation of REAL and SYNTHETIC subsets
            train_dst_mixed_hybrid = data.ConcatDataset([train_dst_synthetic, train_dst_real])
            train_dst_mixed_hybrid_indices = list(np.arange(0, len(train_dst_mixed_hybrid), 1))
            random.Random(opts.random_seed).shuffle(train_dst_mixed_hybrid_indices)
            train_dst_mixed_hybrid = data.Subset(dataset=train_dst_mixed_hybrid, indices=train_dst_mixed_hybrid_indices) # Filter indices

            # for i in range(len(train_dst_mixed_hybrid)):
            #     torchvision.transforms.ToPILImage()(train_dst_mixed_hybrid[i][0]).show()

            # exit(2)
            print('Train dataset size: ', len(train_dst_mixed_hybrid))
            print('\t[Real]: ', len(train_dst_real))
            print('\t[Synthetic]: ', len(train_dst_synthetic))
            print('Val dataset size: ', len(val_dst))
            print('Test dataset size: ', len(test_dst))
            
            train_dst = train_dst_mixed_hybrid

            for idx in train_dst_real_indices:
                if idx in test_dst_indices:
                    print('Index conflict')
                    exit(1)

            # visualize_class_distribution(train_dst_synthetic)
            # visualize_class_distribution(test_dst)
            # exit(1)
            
        else:
            train_dst = HybridDataset(root_path=opts.data_root + '\\' + opts.data_source + '\\train',
                                    input_dir='rgb',
                                    target_dir='semantic_segmentation_mapped',
                                    transform=train_transform,
                                    type=opts.data_source)
        
            # Creation of real, entire-dataset, indices
            train_dst_indices = list(np.arange(0, len(train_dst), 1))
            test_dst_indices = list(np.arange(0, len(test_dst), 1))

            # Shuffling of entire-dataset indices
            temp_indices = list(zip(train_dst_indices, test_dst_indices))
            random.Random(opts.random_seed).shuffle(temp_indices)
            train_dst_indices, test_dst_indices = zip(*temp_indices)
            train_dst_indices, test_dst_indices = list(train_dst_indices), list(test_dst_indices)
             
            # Slicing entire-dataset indices (Train and Test - 500 examples)
            train_dst_indices = train_dst_indices[0:opts.max_train_examples]
            test_dst_indices = test_dst_indices[len(test_dst)-500:len(test_dst)]

            print('Train dataset size: ', len(train_dst_indices))
            print('Val dataset size: ', len(val_dst))
            print('Test dataset size: ', len(test_dst_indices))
            
            # Initializing subsets based on previously computed indices
            train_dst = data.Subset(dataset=train_dst, indices=train_dst_indices)
            test_dst = data.Subset(dataset=test_dst, indices=test_dst_indices)

            for idx in train_dst_indices:
                if idx in test_dst_indices:
                    print('Index conflict')
                    exit(1)

            # visualize_class_distribution(train_dst)
            # visualize_class_distribution(test_dst)
            # exit(3)

    return train_dst, val_dst, test_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
    elif opts.dataset.lower() == 'hybrid_dataset':
        opts.num_classes = 17

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst, test_dst = get_dataset(opts)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    test_loader = data.DataLoader(
        test_dst, batch_size=1, shuffle=True, num_workers=2)
    
    print("Dataset: %s, Train set: %d, Val set: %d, Test set: %d" %
          (opts.dataset, len(train_dst), len(val_dst), len(test_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride, pretrained_backbone=opts.pretrained_backbone)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)

        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)

        print("Model state loaded from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    if not opts.freeze_layers == '':
        layer_names = [str(layer_name) for layer_name in opts.freeze_layers.split(',')]
        
        # Freezing layers for fine-tuning
        for name, param in model.named_parameters():
            for layer_name in layer_names:
                if param.requires_grad and layer_name in name:
                    print("[!] Freezing: ", name)
                    param.requires_grad = False

    model_name_prefix = ''
    if (not opts.mixed_hybrid):
        model_name_prefix = 'checkpoints/%s_%s_%d_os%d_%s_' % (opts.data_source, 
                                                           opts.dataset,
                                                           opts.max_train_examples,
                                                           opts.output_stride,
                                                           'finetune' if opts.freeze_layers != '' else 'pretrain')
    else:
        model_name_prefix = 'checkpoints/mixedhybrid_%s_R%d_S%d_os%d_%s_' % (opts.dataset,
                                                           opts.mixed_hybrid_real_examples,
                                                           opts.mixed_hybrid_synthetic_examples,
                                                           opts.output_stride,
                                                           'finetune' if opts.freeze_layers != '' else 'pretrain')
       
    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        test_score, ret_samples = validate(
            opts=opts, model=model, loader=test_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        
        test_metrics_prefix = opts.ckpt.split('/')[-1].split('.')[0] if opts.ckpt else model_name_prefix
        with open('checkpoints/' + test_metrics_prefix + '_test_metrics.txt', 'w') as f:
            f.write(json.dumps(test_score, indent=4))
        return

    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt(model_name_prefix + 'LATEST.pth')
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt(model_name_prefix + 'BEST.pth')

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                with open(model_name_prefix + 'LATEST' + '_train_metrics.txt', 'w') as f:
                    val_score_text = json.dumps(val_score, indent=4)
                    f.write(val_score_text)

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8) if opts.dataset != 'hybrid_dataset' else HybridDataset.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8) if opts.dataset != 'hybrid_dataset' else HybridDataset.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return
            

if __name__ == '__main__':
    main()
