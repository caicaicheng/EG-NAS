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
import torchvision.datasets as dset
import torch.backends.cudnn as cudnns

from torch.autograd import Variable
from model_search import Network

import time
from cmaes import CMA


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')#50
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--iterations', type=int, default=3, help='T')#e
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--celltype', type=str, default='normal', help='experiment name')
parser.add_argument('--resume', type=str, default='', help='resume from pretrained')#/home/cai/cai_code/Shapley-NAS/search-EXP-20230522-050112/weights_pretrain.pt
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--pop_momentum', type=float, default=0.0, help='momentum for updating shapley')
parser.add_argument('--step_size', type=float, default=0.1, help='step size for updating shapley')
parser.add_argument('--samples', type=int, default=3, help='number of samples for estimation')
parser.add_argument('--mon',type=int,default=0.4,help='')
parser.add_argument('--warm_start_epochs', type=int, default=15,
                    help='Warm start one-shot model before starting architecture updates.')
args = parser.parse_args()

args.save = '../search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10
if args.set=='cifar100':
    CIFAR_CLASSES = 100
def main():
  CUDA_VISIBLE_DEVICES = 1

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  
  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnns.benchmark = True
  torch.manual_seed(args.seed)
  cudnns.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
#   if args.resume:
#       model.load_state_dict(torch.load(args.resume))
#       model.show_arch_parameters()
#       genotype = model.genotype()
#       logging.info('genotype = %s', genotype)


  arch_params = list(map(id, model.arch_parameters()))
  weight_params = filter(lambda p: id(p) not in arch_params,
                         model.parameters())
  optimizer = torch.optim.SGD(
    weight_params,
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  if args.set=='cifar100':
      train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
  else:
      train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    
  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

#
    
  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size // 2,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True,num_workers=8)

  infer_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)


  ops = []
  for cell_type in ['normal','reduce']:
      for edge in range(model.num_edges):
          ops.append(['{}_{}_{}'.format(cell_type, edge, i) for i in
                          range(0, model.num_ops)])
  ops = np.concatenate(ops)

  if args.resume:
      train_epochs = 25
  else:
      pretrain_epochs=0
      train_epochs = args.epochs - pretrain_epochs
  epoch = 0
  best_individual = [1e-3 * torch.randn(model.num_edges, model.num_ops).cuda(),1e-3 * torch.randn(model.num_edges, model.num_ops).cuda()]
  for current_epochs in range(train_epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', current_epochs, lr)
        model.show_arch_parameters()
        genotype = model.genotype()
        logging.info('genotype = %s', genotype)
        
        if current_epochs >= args.warm_start_epochs:
            
            best_normal, best_reduce = explore_architecture(valid_queue, model, criterion,epoch, num_samples=args.samples)
            best_individual = updated_alpha(model, [best_normal, best_reduce], best_individual, momentum=args.pop_momentum, step_size=args.step_size)

            model.show_arch_parameters()
            genotype = model.genotype()
            logging.info('genotype = %s', genotype)
        train_acc, train_obj = train(train_queue,valid_queue, model, criterion,lr, optimizer,current_epochs)
        logging.info('train_acc %f', train_acc)
        valid_acc, valid_obj = infer(infer_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)

        if not args.resume and epoch == pretrain_epochs -1:
            utils.save(model, os.path.join(args.save, 'weights_pretrain.pt'))

        utils.save(model, os.path.join(args.save, 'weights.pt'))

  model.show_arch_parameters()
  genotype = model.genotype()
  logging.info('genotype = %s', genotype)


#Compound Fitness Function
def CFF(model,inputs,target,pop_sample,criterion,ori_pre,args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normal = model.get_projected_weights('normal').to(device)
    reduce = model.get_projected_weights('reduce').to(device)
    
    pop_sample = pop_sample.reshape(2,model.num_edges, model.num_ops)
    normal_weights = pop_sample[0]
    reduce_weights = pop_sample[1]
    
    normal_weights = torch.tensor(normal_weights).to(device)
    normal_weights /= normal_weights.norm(dim=-1,keepdim=True)

    reduce_weights = torch.tensor(reduce_weights).to(device)
    reduce_weights /= reduce_weights.norm(dim=-1,keepdim=True)
    
    cossima = nn.CosineSimilarity(dim=-1)
    similar_nor =  (cossima(normal,normal_weights) + 1 ) / 2
    similar_red =  (cossima(reduce,reduce_weights) + 1 ) / 2
    similar_loss =( similar_nor.mean() + similar_red.mean()) / 2
    
    normal_weights = torch.softmax(normal_weights,dim=-1)
    reduce_weights = torch.softmax(reduce_weights,dim=-1)
    
    # normal_weights = normal_weights - normal_weights.mean(dim=0)
    # reduce_weights = reduce_weights - reduce_weights.mean(dim=0)
    # normal = normal - normal.mean(dim=0)
    # reduce = reduce - reduce.mean(dim=0)
    logits = model(inputs,  weights_dict={'normal': normal_weights,'reduce':reduce_weights})
    prec1,_ = utils.accuracy(logits, target, topk=(1, 5))
    loss = criterion(logits,target)
    if ori_pre <= prec1:
        loss = loss + similar_loss*args.mon
    else:
        loss = loss - similar_loss*args.mon
    
    return loss,normal_weights.cpu().detach().numpy(),reduce_weights.cpu().detach().numpy()


#Evolution Strategy sampling
def explore_architecture(valid_queue, model, criterion, epoch, num_samples):
    """
    Implementation of Evolution Strategy sampling of architecture for performance and similarity evaluation
    """
    ori_normal_weights = model.get_projected_weights('normal')
    normal_weights = 1e-2 * torch.randn_like(ori_normal_weights) + ori_normal_weights
    normal_weights = normal_weights.reshape((model.num_edges, model.num_ops)).cpu().detach().numpy()
    ori_reduce_weights = model.get_projected_weights('reduce')
    reduce_weights = 1e-2 * torch.randn_like(ori_reduce_weights) + ori_reduce_weights
    reduce_weights = reduce_weights.reshape((model.num_edges, model.num_ops)).cpu().detach().numpy()
    
    weights = np.zeros((2,model.num_edges * model.num_ops))
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
        now_sample = np.zeros((2*14*7))
        sima = 1 /( epoch + 1)
        sima = sima if sima >= 0.02 else 0.02
        for j in range(num_samples):
            pop_optimizer = CMA(mean=new_weights,sigma=sima,population_size=50,seed=1)
            solutions = []
            for i in range(pop_optimizer.population_size):
                pop_sample = pop_optimizer.ask()
                pop_value,normal,reduce = CFF(model,x,y,pop_sample,criterion,ori_pre)
                solutions.append((pop_sample,pop_value))
                if i == 0 or pop_value <= all_acc:
                    all_acc = pop_value
                    now_sample = pop_sample
                    now_pop_normal = normal
                    now_pop_reduce = reduce
            pop_optimizer.tell(solutions)
            pop_value,normal,reduce = CFF(model,x,y,pop_sample,criterion,ori_pre)
            if pop_value < all_acc:
                now_sample = pop_sample
                all_acc = pop_value
                now_pop_normal = normal
                now_pop_reduce = reduce     


            reduce_pop_sample = torch.tensor(now_pop_reduce)
            normal_pop_sample = torch.tensor(now_pop_normal)
            normal_weights = torch.softmax(normal_pop_sample,dim=-1)
            reduce_weights = torch.softmax(reduce_pop_sample,dim=-1)
            
            logits = model(x,  weights_dict={'normal': normal_weights,'reduce':reduce_weights})
            prec1,_ = utils.accuracy(logits, y, topk=(1, 5))
            if prec1.item() != 0:
                pop_acc[j] = 1/prec1.item()
                pop_normal[j] = normal_weights.cpu().detach().numpy()
                pop_reduce[j] = reduce_weights.cpu().detach().numpy()
            else:
                pop_acc[j] = 0
                pop_normal[j] = 0
                pop_reduce[j] = 0
        acc = pop_acc[0]
        pop_index = 0
        bad_index = 0
        bad_acc = pop_acc[0]
        for i in range(1,num_samples):
            if pop_acc[i] < acc:
                acc = pop_acc[i]
                pop_index = i
            if pop_acc[i] > bad_acc:
                bad_acc = pop_acc[i]
                bad_index = i
        logging.info("loss = %f", 1/acc)

        shap_normal, shap_reduce = pop_normal[pop_index], pop_reduce[pop_index]
        return shap_normal, shap_reduce


def updated_alpha(model, shap_values, accu_shap_values, momentum=0.0, step_size=0.6):
    assert len(shap_values)==len(model.arch_parameters())

    shap = [torch.from_numpy(shap_values[i]).cuda() for i in range(len(model.arch_parameters()))]

    # for i,params in enumerate(shap):
    #     mean = params.data.mean()
    #     std = params.data.std()
    #     params.data.add_(-mean).div_(std)

    updated_shap = [
        accu_shap_values[i] * momentum \
                    + shap[i] * (1. - momentum)
            for i in range(len(model.arch_parameters()))]

    for i,p in enumerate(model.arch_parameters()):
        p.data.add_((step_size * updated_shap[i]).to(p.device))
    # show the best individual
    return updated_shap

def train(train_queue,valid_queue, model, criterion,lr,optimizer,epochs):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target) 
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):

    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda()
    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':

  main()

