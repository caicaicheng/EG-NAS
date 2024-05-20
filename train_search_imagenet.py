import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
# import torch.backends.cudnn as cudnn
import copy
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model_search_imagenet import Network
from cmaes import CMA

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=80, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--resume', type=str, default='', help='resume from pretrained')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='/tmp/checkpoints/', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--pop_momentum', type=float, default=0.0, help='momentum for updating shapley')
parser.add_argument('--step_size', type=float, default=0.1, help='step size for updating shapley')
parser.add_argument('--samples', type=int, default=5, help='number of samples for estimation')
parser.add_argument('--mon',type=int,default=0.4,help='')
parser.add_argument('--begin', type=int, default=10, help='batch size')
parser.add_argument('--pop_size',type=int,default=50,help='')
parser.add_argument('--tmp_data_dir', type=str, default='../data', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')

args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 1000


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)

    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    # dataset split
    train_data1 = dset.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    train_data2 = dset.ImageFolder(valdir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    valid_data = dset.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    num_train = len(train_data1)
    num_val = len(train_data2)
    print('# images to train network: %d' % num_train)
    print('# images to validate network: %d' % num_val)

    model = Network(args.init_channels, CLASSES, args.layers, criterion)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    start_epoch = 0

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    optimizer_a = torch.optim.Adam(model.module.arch_parameters(),
                                   lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                   weight_decay=args.arch_weight_decay)

    train_queue = torch.utils.data.DataLoader(
        train_data1, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        train_data2, batch_size=args.batch_size // 2, shuffle=True,
        pin_memory=True, num_workers=args.workers)

    infer_queue = torch.utils.data.DataLoader(
        train_data2, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=args.workers)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    if args.resume:
        checkpoint = torch.load(os.path.join(args.resume, 'checkpoint.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])

        model.module.show_arch_parameters()
        genotype = model.module.genotype()
        logging.info('genotype = %s', genotype)

    ops = []
    for cell_type in ['normal', 'reduce']:
        for edge in range(model.module.num_edges):
            ops.append(['{}_{}_{}'.format(cell_type, edge, i) for i in
                        range(0, model.module.num_ops)])
    ops = np.concatenate(ops)
    lr = args.learning_rate
    best_individual = [1e-3 * torch.randn(model.module.num_edges, model.module.num_ops).cuda(),
                   1e-3 * torch.randn(model.module.num_edges, model.module.num_ops).cuda()]

    for epoch in range(start_epoch, args.epochs):
        scheduler.step()
        current_lr = scheduler.get_lr()[0]
        logging.info('Epoch: %d lr: %e', epoch, current_lr)
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)
            print(optimizer)
        genotype = model.module.genotype()
        logging.info('genotype = %s', genotype)

        if epoch >= args.epochs // 2:

            best_normal, best_reduce = explore_architecture(valid_queue, model, criterion,epoch,args)
            best_individual = updated_alpha(model, [best_normal, best_reduce], best_individual, momentum=args.shapley_momentum, step_size=args.step_size)


        train_acc, train_obj = train(train_queue, model, optimizer, criterion)
        logging.info('Train_acc %f', train_acc)

        # validation
        if epoch >= args.epochs-3:#47
            valid_acc, valid_obj = infer(infer_queue, model, criterion)
            logging.info('Valid_acc %f', valid_acc)

        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'alpha': model.module.arch_parameters()
        }, False, args.save)

    genotype = model.module.genotype()
    logging.info('genotype = %s', genotype)


#Evolution Strategy sampling
def explore_architecture(valid_queue, model, criterion, epoch, args):
    """
    Implementation of Evolution Strategy sampling of architecture for performance and similarity evaluation

    """
    num_samples = args.num_samples
    normal_weights = model.get_projected_weights('normal')
    
    normal_weights = 1e-2 * torch.rand_like(normal_weights) + normal_weights
    normal_weights = normal_weights.reshape((model.num_edges, model.num_ops)).cpu().detach().numpy()
    reduce_weights = model.get_projected_weights('reduce')
    reduce_weights = 1e-2 * torch.rand_like(reduce_weights) + reduce_weights
    reduce_weights = reduce_weights.reshape((model.num_edges, model.num_ops)).cpu().detach().numpy()
    
    weights = np.zeros((2,model.num_edges*model.num_ops))
    weights[0] = normal_weights.reshape(-1)
    weights[1] = reduce_weights.reshape(-1)
    new_weights = weights.reshape(-1)
    
    with torch.no_grad():
        x, y = next(iter(valid_queue))
        x, y = x.cuda(), y.cuda(non_blocking=True)
        log_X = model(x)
        ori_pre,_ = utils.accuracy(log_X, y, topk=(1, 5))
        
        pop_normal = np.zeros((num_samples,model.num_edges,model.num_ops))
        pop_reduce = np.zeros((num_samples,model.num_edges,model.num_ops))
        pop_acc = np.zeros(num_samples)
        
        now_pop_normal = np.zeros((model.num_edges,model.num_ops))
        
        now_pop_reduce = np.zeros((model.num_edges,model.num_ops))
        
        all_acc =  0
        sima = 1 /( epoch + 1)
        sima = sima if sima >= 0.02 else 0.02
        for j in range(num_samples):
            pop_optimizer = CMA(mean=new_weights,sigma=sima,population_size=args.pop_size,seed=args.seed)
            solutions = []
            for i in range(pop_optimizer.population_size):
                pop_sample = pop_optimizer.ask()
                pop_value,normal,reduce = CFF(model,x,y,pop_sample,criterion,ori_pre,args)
                solutions.append((pop_sample,pop_value))
                if i == 0 or pop_value <= all_acc:
                    all_acc = pop_value
                    now_sample = pop_sample
                    now_pop_normal = normal
                    now_pop_reduce = reduce
                    
            pop_optimizer.tell(solutions)
            pop_value,normal,reduce = CFF(model,x,y,pop_sample,criterion,ori_pre,args)
            if pop_value < all_acc:
                now_sample = pop_sample
                all_acc = pop_value
                now_pop_normal = normal
                now_pop_reduce = reduce     


            reduce_pop_sample = torch.tensor(now_pop_reduce)
            normal_pop_sample = torch.tensor(now_pop_normal)
            
            normal_weights = normal_pop_sample
            reduce_weights = reduce_pop_sample
            
            logits = model(x,  weights_dict={'normal': normal_weights,'reduce':reduce_weights})
            prec1,_ = utils.accuracy(logits, y, topk=(1, 5))
            
            pop_acc[j] = 1/prec1.item()
            pop_normal[j] = normal_weights.cpu().detach().numpy()
            pop_reduce[j] = reduce_weights.cpu().detach().numpy()
        acc = pop_acc[0]
        pop_index = 0
        for i in range(1,num_samples):
            if pop_acc[i] < acc:
                acc = pop_acc[i]
                pop_index = i
        logging.info("loss = %f", 1/acc)

        shap_normal, shap_reduce = pop_normal[pop_index], pop_reduce[pop_index]
        return shap_normal, shap_reduce


#Compound Fitness Function
def CFF(model,inputs,target,pop_sample,criterion,ori_pre,args):
    pop_sample = pop_sample.reshape(2,model.num_edges, model.num_ops)
    normal_weights = pop_sample[0]
    reduce_weights = pop_sample[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normal_weights = torch.tensor(normal_weights).to(device)
    reduce_weights = torch.tensor(reduce_weights).to(device)
    
    normal_weights = torch.softmax(normal_weights,dim=-1)
    reduce_weights = torch.softmax(reduce_weights,dim=-1)
    
    normal = model.get_projected_weights('normal').to(device)
    reduce = model.get_projected_weights('reduce').to(device)
    
    
    cossima = nn.CosineSimilarity(dim=-1)
    similar_nor =  (cossima(normal,normal_weights) + 1 ) / 2
    similar_red =  (cossima(reduce,reduce_weights) + 1 ) / 2
    similar_loss = (similar_nor.mean() + similar_red.mean()) / 2

    
    logits = model(inputs,  weights_dict={'normal': normal_weights,'reduce':reduce_weights})
    prec1,_ = utils.accuracy(logits, target, topk=(1, 5))
    loss = criterion(logits,target)
    
    if ori_pre <= prec1:
        loss = loss + similar_loss*args.mon
    else:
        loss = loss - similar_loss*args.mon
    
    return loss,normal_weights.cpu().detach().numpy(),reduce_weights.cpu().detach().numpy()


def updated_alpha(model, pop_values, perturbation_factors, momentum=0.0, step_size=0.1):
    assert len(pop_values) == len(model.module.arch_parameters())

    shap = [torch.from_numpy(pop_values[i]).cuda() for i in range(len(model.module.arch_parameters()))]

    # for i, params in enumerate(shap):
    #     mean = params.data.mean()
    #     std = params.data.std()
    #     params.data.add_(-mean).div_(std)

    updated_individual = [
        perturbation_factors[i] * momentum \
        + shap[i] * (1. - momentum)
        for i in range(len(model.module.arch_parameters()))]

    for i, p in enumerate(model.module.arch_parameters()):
        p.data.add_((step_size * updated_individual[i]).to(p.device))
    # show the best individual
    return updated_individual


def train(train_queue, model, optimizer, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()

