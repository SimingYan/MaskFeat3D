import torch
import torch.nn as nn
from tools import builder
from lib import misc, dist_utils
import time
from lib.logger import *
from lib.AverageMeter import AverageMeter

from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms


class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict

def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]

def run_net(args, config, train_writer=None, val_writer=None):
    transform_list = []
    if config.rotate:
        transform_list.append(data_transforms.PointcloudNormalRotate())
    if config.jitter:
        transform_list.append(data_transforms.PointcloudNormalJitter())
    if config.scale_translate:
        transform_list.append(data_transforms.PointcloudSingleScaleAndTranslate())
    if config.scale:
        transform_list.append(data_transforms.PointcloudSingleScale())

    if len(transform_list) > 0:
        train_transforms = transforms.Compose(transform_list)
    else:
        train_transforms = None

    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder_update(args, config.dataset.train), \
                                                                builder.dataset_builder_update(args, config.dataset.val)
    (_, extra_train_dataloader)  = builder.dataset_builder_update(args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)
    (_, extra_test_dataloader)  = builder.dataset_builder_update(args, config.dataset.extra_test) if config.dataset.get('extra_test') else (None, None)
    
    # build model
    base_model = builder.model_builder(config.model)

    print(base_model)

    if args.use_gpu:
        base_model.to(args.local_rank)
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)
    
    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
        #base_model = base_model.cuda()

    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    base_model.zero_grad()
    
    train_dataloader_iter = train_dataloader.__iter__()
    iter_per_epoch = len(train_dataloader)

    for epoch in range(start_epoch, config.max_epoch + 1):
        
        #if args.distributed:
        #    train_sampler.set_epoch(epoch)
        
        base_model.train()  # set model to training mode

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])

        num_iter = 0

        n_batches = len(train_dataloader)
       
        for idx in range(iter_per_epoch):
            taxonomy_ids, model_ids, data = train_dataloader_iter.next()

            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME

            if dataset_name == 'ShapeNetFeat':
                points = data['pc'].cuda()
                normal = data['normal'].cuda()
                sv = data['sv'].cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            assert points.size(1) == npoints
            
            if train_transforms:
                data = {'pc': points, 'normal': normal}
                data = train_transforms(data)
                points = data['pc']
                normal = data['normal']
            
            loss, loss_metrics = base_model(points, normal, sv, vis=False, noaug=False)

            loss.backward()
            
            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()
            
            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss.item()*1000])
            else:
                losses.update([loss.item()*1000])

            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

                if 'normal_loss' in loss_metrics:
                    train_writer.add_scalar('Loss/Batch/Normal_Loss', loss_metrics['normal_loss'].item(), n_itr)
                if 'sv_loss' in loss_metrics:
                    train_writer.add_scalar('Loss/Batch/SV_Loss', loss_metrics['sv_loss'].item(), n_itr)
                if 'pc_loss' in loss_metrics:
                    train_writer.add_scalar('Loss/Batch/PC_Loss', loss_metrics['pc_loss'].item(), n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
            

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
             optimizer.param_groups[0]['lr']), logger = logger)
        

        if epoch % args.val_freq == 0 and config.model.get('test') != True and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, extra_train_dataloader, extra_test_dataloader, epoch, val_writer, args, config, logger=logger)
    
            # Save ckeckpoints
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-best-{epoch:03d}', args, logger = logger)
        
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        
        if epoch % args.save_freq == 0:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args,
                                    logger=logger)

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_features = []
    test_label = []

    train_features = []
    train_label = []
    npoints = config.dataset.train.others.npoints

    with torch.no_grad():
        extra_train_dataloader_iter = extra_train_dataloader.__iter__()
        iter_per_epoch = len(extra_train_dataloader)
        
        #for idx, (taxonomy_ids, model_ids, data) in enumerate(extra_train_dataloader):
        for idx in range(iter_per_epoch):
            taxonomy_ids, model_ids, data = extra_train_dataloader_iter.next()
            
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            assert points.size(1) == npoints
            
            feature = base_model(points, None, None, noaug=True)
            target = label.view(-1)

            train_features.append(feature.detach())
            train_label.append(target.detach())
        
        #for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
        test_dataloader_iter = test_dataloader.__iter__()
        iter_per_epoch = len(test_dataloader)
        for idx in range(iter_per_epoch):
            taxonomy_ids, model_ids, data = test_dataloader_iter.next()
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)
            assert points.size(1) == npoints
            
            feature = base_model(points, None, None, noaug=True)
            target = label.view(-1)

            test_features.append(feature.detach())
            test_label.append(target.detach())


        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(), test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch,svm_acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', svm_acc, epoch)

    return Acc_Metric(svm_acc)
