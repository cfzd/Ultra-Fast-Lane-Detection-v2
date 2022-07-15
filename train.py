import torch, os, datetime


from utils.dist_utils import dist_print, dist_tqdm, synchronize
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import update_metrics, reset_metrics

from utils.common import calc_loss, get_model, get_train_loader, inference, merge_config, save_model, cp_projects
from utils.common import get_work_dir, get_logger

import time
from evaluation.eval_wrapper import eval_lane

def train(net, data_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, dataset):
    net.train()
    progress_bar = dist_tqdm(train_loader)
    for b_idx, data_label in enumerate(progress_bar):
        global_step = epoch * len(data_loader) + b_idx

        results = inference(net, data_label, dataset)

        loss = calc_loss(loss_dict, results, logger, global_step, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)


        if global_step % 20 == 0:
            reset_metrics(metric_dict)
            update_metrics(metric_dict, results)
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
            logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

            if hasattr(progress_bar,'set_postfix'):
                kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
                new_kwargs = {}
                for k,v in kwargs.items():
                    if 'lane' in k:
                        continue
                    new_kwargs[k] = v
                progress_bar.set_postfix(loss = '%.3f' % float(loss), 
                                        **new_kwargs)
        

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    if args.local_rank == 0:
        work_dir = get_work_dir(cfg)

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

        if args.local_rank == 0:
            with open('.work_dir_tmp_file.txt', 'w') as f:
                f.write(work_dir)
        else:
            while not os.path.exists('.work_dir_tmp_file.txt'):
                time.sleep(0.1)
            with open('.work_dir_tmp_file.txt', 'r') as f:
                work_dir = f.read().strip()

    synchronize()
    cfg.test_work_dir = work_dir
    cfg.distributed = distributed

    if args.local_rank == 0:
        os.system('rm .work_dir_tmp_file.txt')
    
    dist_print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')
    dist_print(cfg)
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide', '34fca']

    train_loader = get_train_loader(cfg)
    net = get_model(cfg)

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank])
    optimizer = get_optimizer(net, cfg)

    if cfg.finetune is not None:
        dist_print('finetune from ', cfg.finetune)
        state_all = torch.load(cfg.finetune)['model']
        state_clip = {}  # only use backbone parameters
        for k,v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)
    if cfg.resume is not None:
        dist_print('==> Resume model from ' + cfg.resume)
        resume_dict = torch.load(cfg.resume, map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1
    else:
        resume_epoch = 0

    scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    dist_print(len(train_loader))
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)
    logger = get_logger(work_dir, cfg)
    # cp_projects(cfg.auto_backup, work_dir)
    max_res = 0
    res = None
    for epoch in range(resume_epoch, cfg.epoch):

        train(net, train_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, cfg.dataset)
        train_loader.reset()

        res = eval_lane(net, cfg, ep = epoch, logger = logger)

        if res is not None and res > max_res:
            max_res = res
            save_model(net, optimizer, epoch, work_dir, distributed)
        logger.add_scalar('CuEval/X',max_res,global_step = epoch)

    logger.close()
