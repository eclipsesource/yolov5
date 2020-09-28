import argparse
import sys
import test  # import test.py to get mAP after each epoch

import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from models.yolo import Model
from utils import google_utils
from utils.datasets import *
from utils.utils import *

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
    print('Using mixed precision training: https://github.com/NVIDIA/apex')
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

# Hyperparameters
hyp = {'optimizer': 'SGD',  # ['adam', 'SGD', None] if none, default is SGD
       'lr0': 0.000706,  # initial learning rate (SGD=1E-2, Adam=1E-3)
       'momentum': 0.692,  # SGD momentum/Adam beta1
       'weight_decay': 0.000277,  # optimizer weight decay
       'giou': 0.374,  # giou loss gain
       'cls': 0.342,  # cls loss gain
       'cls_pw': 0.717,  # cls BCELoss positive_weight
       'obj': 52.8,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.674,  # iou training threshold
       'anchor_t': 3.1,  # anchor-multiple threshold
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0,  # image HSV-Value augmentation (fraction)
       'degrees': 0.0,  # image rotation (+/- deg)
       'translate': 0.0,  # image translation (+/- fraction)
       'scale': 0.652,  # image scale (+/- gain)
       'shear': 0.0}  # image shear (+/- deg)

def prepare_saving_paths(tb_writer):
    log_dir = tb_writer.log_dir if tb_writer else 'runs/evolution'
    wdir = log_dir + os.sep + 'weights'  # weights directory runs/exp*/weights/
    os.makedirs(wdir, exist_ok=True)
    return {
        'log_dir': log_dir,
        'wdir': wdir,
        'last': wdir + os.sep + 'last.pt',
        'best': wdir + os.sep + 'best.pt',
        'results': log_dir + os.sep + 'results.txt',
        'hyperparams': log_dir + os.sep + 'hyp.yaml',
        'opt': log_dir + os.sep + 'opt.yaml'
    }

def save_params_to_yaml(params, saving_path: str):
    with open(saving_path, 'w') as f:
        yaml.dump(params, f, sort_keys=False)

def get_data_info(data_path, single_cls):
    with open(data_path) as f: # Load train and val data path
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    data_dict['nc'], data_dict['names'] = (1, ['item']) if single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(data_dict['names']) == data_dict['nc'], '%g names found for nc=%g dataset in %s' % (len(data_dict['names']), data_dict['nc'], data_path)
    return data_dict

def prepare_optimizer(optimizer, lr0, momentum, weight_decay, model):
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        if v.requires_grad:
            if '.bias' in k:
                pg2.append(v)  # biases
            elif '.weight' in k and '.bn' not in k:
                pg1.append(v)  # apply weight decay
            else:
                pg0.append(v)  # all else
    if optimizer == 'adam':  # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
        optimizer = optim.Adam(pg0, lr=lr0, betas=(momentum, 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=lr0, momentum=momentum, nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': weight_decay})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    return optimizer

def load_weight(model, ckpt):
    try:
        ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items() if model.state_dict()[k].shape == v.shape}  # Load weight from file while layer matches to FP32, filter
        model.load_state_dict(ckpt['model'], strict=False)
    except KeyError as e:
        s = "Saved weight is not compatible with model. This may be due to model differences or saved weight may be out of date. " \
            "Please delete or update saved weight and try again, or use --weights '' to train from scratch."
        raise KeyError(s) from e
    return model

def load_ckpt(weight_path):
    if not weight_path:
        return
    if not opt.weights.endswith('.pt'):  # pytorch format
        sys.exit(f'Cannot find weight file at: {weight_path}')
    ckpt = torch.load(weight_path, map_location=device)  # load checkpoint
    return ckpt

def train(hyp, opt):
    path_dict = prepare_saving_paths(tb_writer)
    save_params_to_yaml(hyp, path_dict['hyperparams']) # Save run settings to yaml files
    save_params_to_yaml(vars(opt), path_dict['opt'])
    # Configure
    init_seeds(1)
    data_dict = get_data_info(opt.data, opt.single_cls)
    # Create model
    model = Model(opt.cfg, nc=data_dict['nc']).to(device)
    # Remove previous results
    for f in glob.glob('*_batch*.jpg') + glob.glob(path_dict['results']): # base path is *yolov5/ ??????????????????????????????????????????????????????????????///
        os.remove(f)
    # Image sizes
    gs = int(max(model.stride))  # grid size (max stride) what are strides???? [32, 16, 8]
    imgsz = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
    # Optimizer
    accumulate = max(round(opt.nbs / opt.batch_size), 1) # accumulate loss before optimizing
    optimizer = prepare_optimizer(hyp['optimizer'], hyp['lr0'], hyp['momentum'], hyp['weight_decay'] * opt.batch_size * accumulate / opt.nbs, model)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / opt.epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs, save_dir=log_dir)
    # Load Model weight
    google_utils.attempt_download(opt.weights)
    start_epoch, best_fitness = 0, 0.0
    ckpt = load_ckpt(opt.weights)
    if ckpt:  # pytorch format
        model = load_weight(model=model, ckpt=ckpt)
         # load optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
        # load results
        if ckpt.get('training_results') is not None:
            with open(path_dict['results'], 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt
        # epochs
        start_epoch = ckpt['epoch'] + 1
        if epochs < start_epoch:
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                    (opt.weights, ckpt['epoch'], opt.epochs))
            opt.epochs += ckpt['epoch']  # finetune additional epochs
    # Mixed precision training https://github.com/NVIDIA/apex ??????????????????????????????
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
    # Distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and dist.is_available():
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:9999', world_size=1, rank=0) # distributed backend# init method# number of nodes# node rank
        model = torch.nn.parallel.DistributedDataParallel(model)
    # Trainloader
    dataloader, dataset = create_dataloader(data_dict['train'], imgsz, opt.batch_size, gs, opt, hyp=hyp, augment=False, cache=opt.cache_images, rect=opt.rect)
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < data_dict['nc'], 'Label class %g exceeds nc=%g in %s. Correct your labels or your model.' % (mlc, data_dict['nc'], opt.cfg)
    # Testloader
    testloader, _ = create_dataloader(data_dict['val'], imgsz, opt.batch_size, gs, opt, hyp=hyp, augment=False, cache=opt.cache_images, rect=True)
    # Model parameters
    hyp['cls'] *= data_dict['nc'] / 4.  # scale coco-tuned class loss gain to current dataset 
    model.nc = data_dict['nc']  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset.labels, data_dict['nc']).to(device)  # attach class weights
    model.names = data_dict['names']
    # Class frequency
    labels = np.concatenate(dataset.labels, 0)
    c = torch.tensor(labels[:, 0])  # classes
    # cf = torch.bincount(c.long(), minlength=nc) + 1.
    # model._initialize_biases(cf.to(device))
    plot_labels(labels, save_dir=path_dict['log_dir'])
    if tb_writer:
        # tb_writer.add_hparams(hyp, {})  # causes duplicate https://github.com/ultralytics/yolov5/pull/384
        tb_writer.add_histogram('classes', c, 0)
    # Check anchors
    if not opt.noautoanchor:
        check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
    # Exponential moving average
    ema = torch_utils.ModelEMA(model)
    # Start training
    t0 = time.time()
    nw = max(3 * nb, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
    maps = np.zeros(data_dict['nc'])  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    scheduler.last_epoch = start_epoch - 1  # do not move
    print(f'Image sizes {imgsz}')
    print('Using %g dataloader workers' % dataloader.num_workers)
    print('Starting training for %g epochs...' % opt.epochs)
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, opt.epochs):  # epoch ------------------------------------------------------------------
        model.train()
        # Update image weights (optional)
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(dataset.labels, nc=data_dict['nc'], class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, opt.nbs / opt.batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            pred = model(imgs)

            # Loss
            loss, loss_items = compute_loss(pred, targets.to(device), model)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Backward
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Optimize
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                '%g/%g' % (epoch, opt.epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)

            # Plot
            if ni < 3:
                f = str(Path(path_dict['log_dir']) / ('train_batch%g.jpg' % ni))  # filename
                result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                if tb_writer and result is not None:
                    tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        scheduler.step()

        # mAP
        ema.update_attr(model, include=['md', 'nc', 'hyp', 'gr', 'names', 'stride'])
        final_epoch = epoch + 1 == opt.epochs
        if not opt.notest or final_epoch:  # Calculate mAP
            results, maps, _ = test.test(opt.data,
                                             batch_size=opt.batch_size,
                                             imgsz=imgsz,
                                             save_json=final_epoch and opt.data.endswith(os.sep + 'coco.yaml'),
                                             model=ema.ema,
                                             single_cls=opt.single_cls,
                                             dataloader=testloader,
                                             save_dir=path_dict['log_dir'])
        # Write
        with open(path_dict['results'], 'a') as f:
            f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        if len(opt.name) and opt.bucket:
            os.system('gsutil cp %s gs://%s/results/results%s.txt' % (path_dict['results'], opt.bucket, opt.name))

        # Tensorboard
        if tb_writer:
            tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
            for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                tb_writer.add_scalar(tag, x, epoch)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi

        # Save model
        save = (not opt.nosave) or (final_epoch and not opt.evolve)
        if save:
            with open(path_dict['results'], 'r') as f:  # create checkpoint
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': f.read(),
                        'model': ema.ema,
                        'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last, best and delete
            torch.save(ckpt, path_dict['last'])
            if (best_fitness == fi) and not final_epoch:
                torch.save(ckpt, path_dict['best'])
            del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    # Strip optimizers
    n = ('_' if len(opt.name) and not opt.name.isnumeric() else '') + opt.name
    fresults, flast, fbest = 'results%s.txt' % n, path_dict['wdir'] + 'last%s.pt' % n, path_dict['wdir'] + 'best%s.pt' % n
    for f1, f2 in zip([path_dict['wdir'] + 'last.pt', path_dict['wdir'] + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
        if os.path.exists(f1):
            os.rename(f1, f2)  # rename
            ispt = f2.endswith('.pt')  # is *.pt
            strip_optimizer(f2) if ispt else None  # strip optimizer
            os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload

    # Finish
    if not opt.evolve:
        plot_results(save_dir=path_dict['log_dir'])  # save as results.png
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if device.type != 'cpu' and torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    return results

def regularize_img_shape(image_shape):
    if not isinstance(image_shape, list): # make sure size is a two elements list
        opt.img_size = [opt.img_size]
    if len(image_shape) is 1:
        return image_shape * 2
    if len(image_shape) > 2:
        sys.exit(f'Image shape cannot be a list larger than two. Got input image shape {image_shape}')
    return image_shape

def print_dict(dictionary: dict):
    for key, value in dictionary.items():
        print(key, ' : ', value)
    print('-' * 50)


if __name__ == '__main__':
    check_git_status()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path') # backbone
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path') # data
    parser.add_argument('--hyp', type=str, default='', help='hyp.yaml path (optional)') # hyperparameters
    parser.add_argument('--epochs', type=int, default=300) # epochs
    parser.add_argument('--batch-size', type=int, default=16) # batch size
    parser.add_argument('--nbs', type=int, default=16) # batch size
    parser.add_argument('--img-size', nargs='+', type=int, default=[1632, 1040], help='training and testing image size. If only one number is given, the image will ') # image size
    parser.add_argument('--rect', action='store_false', help='rectangular training') # do we wanna resize the training images as well or only use the raw image shape
    parser.add_argument('--resume', nargs='?', const='get_last', default='get_last', help='resume from given path/to/last.pt, or most recent run if blank.') # resume from pre training results
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint') # ?
    parser.add_argument('--notest', action='store_true', help='only test final epoch') # ?
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check') # ?
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters') # ?
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket') # ?
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='', help='initial weights path') # want to resume from customized weight
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied') # rename result file
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu') # choose training device
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%') # ?
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset') # If do nms seperately on each class. 'store_true' means False.
    opt = parser.parse_args()
    print('Customized arguments:')
    print_dict(vars(opt))
    last = get_latest_run() if opt.resume == 'get_last' else opt.resume  # resume from most recent run
    if last and not opt.weights:
        print(f'Resuming training from {last} \n' + '-' * 50)
    opt.weights = last if opt.resume and not opt.weights else opt.weights
    opt.cfg = check_file(opt.cfg)  # check existance of backbone
    opt.data = check_file(opt.data)  # check existance of data
    if opt.hyp: # check existance of hyperparameter file if passed path into opt.hyp
        opt.hyp = check_file(opt.hyp)
        with open(opt.hyp) as f:
            hyp.update(yaml.load(f, Loader=yaml.FullLoader))  # update hyps
    opt.img_size = regularize_img_shape(opt.img_size)
    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size) # define devices
    if device.type == 'cpu':
        mixed_precision = False
    print('Hyperparameters are: ')
    print_dict(hyp) 
    # Train
    if not opt.evolve:
        tb_writer = SummaryWriter(log_dir=increment_dir('runs/exp', opt.name))
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/\n'  + '-' * 50)
        train(hyp, opt)
    # Evolve hyperparameters (optional)
    else:
        tb_writer = None
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists
        keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
        for _ in range(300):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination
                # Mutate
                mp, s = 0.9, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 0, 1, 1, 1, 1, 1, 1, 1])  # gains
                ng = len(g)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(list(hyp.keys())[1:]):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate
            # Clip to limits
            #keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                hyp[k] = float(np.clip(hyp[k], v[0], v[1]))

            # Train mutation
            results = train(hyp.copy(), opt)
            # Write mutation results
            print_mutation(hyp, results, opt.bucket)

            # Plot results
            # plot_evolution_results(hyp)
