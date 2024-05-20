from ast import Param
from cProfile import label
from venv import create
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import random

def _concat(xs):
   #把x先拉成一行，然后把所有的x摞起来，变成n行
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.AdamW(self.model.arch_parameters(),#架构参数优化函数
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    self.loss = nn.CrossEntropyLoss().cuda()

   #compute w' / w-ξdwLtrain(w,a)
  def _compute_unrolled_model(self, input, target, eta, network_optimizer,iterations):
    #对ω 参数，Ltrain loss
    #theta = theta + v + weight_decay * theta  这里theta就是我们要更新的参数w
    #w - ξ*dwLtrain(w,a)    
    loss = self.model._loss(input, target)
 
    #n个参数变成n行，需更新的参数theta
    theta = _concat(self.model.parameters()).data#299578
    theta.requires_grad_(True)
    try:
        #增加动量 
        #momentum*v,用的就是Network进行w更新的momentum
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
        #不加动量
      moment = torch.zeros_like(theta)
    #前面的是loss对参数theta求梯度，self.network_weight_decay*thetas就是正则项
    dd = torch.autograd.grad(loss,self.model.parameters())
    #x = _concat(dd).numel()
    for index in range(iterations):
      # ran_var = random.randint(0,x)
      # var = _concat(dd)[ran_var] 
      dtheta = _concat(dd) + self.network_weight_decay*theta
      theta = theta.sub(eta,moment+dtheta)
    unrolled_model = self._construct_model_from_theta(theta)
    #w-ξ*dwLtrain(w,a) 用下列的函数更新网络结构里面的参数，这里传进去的数就已经是更新完的w了
    #theta.sub(eta, moment+dtheta): theta = theta - eta*(moment+dtheta)
    #也就是传进去的就是w = w - eta * (moment + dw)
    
    return unrolled_model
  def mlc_loss(self, arch_param):
    y_pred_neg = _concat(arch_param)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    aux_loss = torch.mean(neg_loss)
    return aux_loss

#实现学习率优化 
  def step(self,input_train, target_train, input_valid, target_valid, eta, network_optimizer,epoch,args):   
    #清楚之前的参数值梯度   
    self.optimizer.zero_grad()
    if args.unrolled:#使用一阶还是二阶 
          #利用公式对α进行update 
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer,epoch,args)
    else:
          #普通优化，交替优化，仅优化alpha，简单求导
        self._backward_step(input_valid, target_valid,epoch)
    self.optimizer.step()#optimizer 存了alpha参数的指针
    
    
#实现架构参数的更新 反向传播,计算梯度  简单来说就是更新a
  def _backward_step(self, input_valid, target_valid,epoch):
    # weights = 0 + 25*epoch/50
    # ssr_normal = self.mlc_loss(self.model.arch_parameters())
    #loss = self.model._loss(input_valid, target_valid) + weights*ssr_normal
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

    #对α进行update
  def _backward_step_unrolled(self,input_train, target_train, input_valid, target_valid, eta, network_optimizer,epoch,args):
    #计算w' = w - ξdwLtrain(w,α)
    #对做了一次更新后的w的unrolled_model求验证集的损失，Lval，以用来对α进行更新
    #unrolled_model ---> w'/w - w-ξ*dwLtrain(w,a)
    
    #unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer,args.iterations)
    #得到w'的损失函数---unrolled_LOSS 就是w' = w-ξ*dwLtrain(w,a)的损失函数
    unrolled_loss = self.model._loss(input_valid, target_valid)
    unrolled_loss.backward()#compute gradient 
    dalpha = [v.grad for v in self.model.arch_parameters()]
    #stcobio
    #dwLval(w',α)
    loss = self.model._loss(input_valid,target_valid)
    v_0 = torch.autograd.grad(loss,self.model.parameters())
    v_0 = _concat(v_0)
    vector = [v.grad.data for v in self.model.parameters()]
    #v_Q =self._compute_vQ_softmax(input_train,target_train,eta,v_0,vector,args.hessian_q)#(299578,1)
    v_Q =self._compute_v_Q_softmax(input_train,target_train,eta,v_0,vector,args.hessian_q)#(299578,1)
    #v_Q = self._compute_v2_Q(input_train,target_train,eta,v_0,vector,args.hessian_q)#太慢了
    dawLtrain = self._compute_dawLtrain1(input_train,target_train,input_valid,target_valid,v_Q)
    

    #要确保这里出来的还是要是4个组，也就是hyparam不能拉直计算
    #update_outputer = (-1) * dawLtrain #这个地方可以建立0组进行减法
    
    #update_outputer = torch.squeeze(dawLtrain)
    #,update_outputer还要和dalpha同属性和尺度，也就是：[14,8],[14,8],[14],[14]
    #这个地方是分为4组，每一组都update 因此的话这里的update_outputer也要是4组,也就是list[tensor]的形式
    
    #tuple = [( x - eta * y ) for x,y in zip(dalpha,dawLtrain)] 


    #Compute expression (8) from paper
    #implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

       # Compute expression (7) from paper

    with torch.no_grad():
      for g,ig in zip (dalpha,dawLtrain):
        g.data.sub_(eta,ig)
        #g = g - eta*ig
    for v,g in zip(self.model.arch_parameters(),dalpha):
      if v.grad is None:
          v.grad = Variable(g)
      else:#将g.data 复制粘贴到v.grad.data上
        v.grad.data.copy_(g)
    #
  
        

  #对应optimizer.step()，对新建的模型的参数进行更新
  def _construct_model_from_theta(self, theta):
        #新建network，copy alpha参数
    model_new = self.model.new()
    model_dict = self.model.state_dict()
    #按照之前的大小，copy theta参数
    params, offset = {}, 0
    for k, v in self.model.named_parameters():
        #k是参数的名字，v是参数
      v_length = np.prod(v.size())#参数的长度
      params[k] = theta[offset: offset+v_length].view(v.size())#将参数k的值更新为theta对应的值
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
  
    model_new.load_state_dict(model_dict)
    return model_new.cuda()#返回 参数更新为做一次反向传播的值 的模型

  def _compute_dawLtrain(self,input_train,target_train,input_valid,target_valid,v_Q):
    #hyparam = self.model.arch_parameters()#tuple -> 4 

    loss = self.model._loss(input_train,target_train)
    loss.requires_grad_(True)
    dLtrain_w = torch.autograd.grad(loss, self.model.parameters(),create_graph=True)#,create_graph=True
    daLtrain_w = _concat(dLtrain_w)
    #v_Q.requires_grad_(True)
#    dav_Q = torch.autograd.grad(v_Q,self.model.arch_parameters())
   # aaa = [(v * w) for v,w in zip(dLtrain_w,dav_Q)]
    hyparam = self.model.arch_parameters()
    #z = _concat(z)
    Likehood_function  = torch.matmul(daLtrain_w,v_Q)
    Likehood_function = torch.squeeze(Likehood_function)

    #因为对hyparam求导，所以出来的是hyparam的格式
    dawLtrain = torch.autograd.grad(Likehood_function,hyparam,allow_unused=True)

    # val_loss = self.model._loss(input_valid,target_valid)
    # dF = torch.autograd.grad(val_loss,self.model.arch_parameters())

    #final_dawLtrain =  [(v - w) for v,w in zip(dF,dawLtrain)]
    
    return dawLtrain
  
  def _compute_dawLtrain1(self,input_train,target_train,input_val,target_val,v_Q):
    r = 1e-2
    R = r / _concat(v_Q).norm()

    
    for p,v in zip(self.model.parameters(),v_Q):
      p.data.add_(R,v)
    loss = self.model._loss(input_train,target_train)
    grads_p = torch.autograd.grad(loss,self.model.arch_parameters())

    for p,v in zip(self.model.parameters(),v_Q):
        p.data.sub_(2*R,v)
        
    loss = self.model._loss(input_train,target_train)
    grads_v = torch.autograd.grad(loss,self.model.arch_parameters())

    for p,v in zip (self.model.parameters(),v_Q):
        p.data.add_(R,v)

    dawLtrain = [(x-y).div_(2*R) for x, y in zip(grads_p, grads_v)]
        
    
    # loss = self.model._loss(input_val,target_val)
    # grads_f = torch.autograd.grad(loss,self.model.arch_parameters())
    #Likehood_function = [(x-y) for x, y in zip(grads_f, dawLtrain)]

    return dawLtrain  
  
  def infine_different(self,input,target,eta,G):
    r = 1e-2
    R = r / _concat(G).norm()
    
    for p,v in zip(self.model.parameters(),G):
      p.data.add_(R,v)
    loss = self.model._loss(input,target)
    grads_p = torch.autograd.grad(loss,self.model.parameters())
    for p,v in zip(self.model.parameters(),G):
      p.data.sub_(2*R,v)   
    loss = self.model._loss(input,target)
    grads_v = torch.autograd.grad(loss,self.model.parameters())  
    
    for p,v in zip(self.model.parameters(),G):
      p.data.add_(R,v)
      
    dw = [(x-y).div_(2*R) for x, y in zip(grads_p, grads_v)]
    dw = [(y - r * x) for x,y in zip(G,dw)]
    return dw
      
    

  def _compute_v_Q(self,input,target,eta,v_0,vector,hessian_q):
    z_list= []
    params = self.model.parameters()
    loss = self.model._loss(input,target)
    loss.requires_grad_(True)
    Gw_gradient = torch.autograd.grad(loss, self.model.parameters())
    G_gradient = [(x-eta*y) for x, y in zip(self.model.parameters(), Gw_gradient)]
    G_gradient = _concat(G_gradient)
    v1 = v_0
    v1 = _concat(v1)
    v_0 = torch.unsqueeze(torch.reshape(v_0, [-1]), 1).detach()
    #Gw_gradient = torch.tensor(Gw_gradient) #solve q2 tuple->tensor
    
    for _ in range(hessian_q):
      Jacobian = torch.matmul(G_gradient,v_0)
      #x
      #q:grad can be implicitly created only for scalar outputs
      #因为auto出来的要是标量，而我这里要是tensor，所以要加上一个grad_outputs = torch.ones_like('tensor_variable')       
      v_new = torch.autograd.grad(Jacobian, self.model.parameters(),retain_graph=True)
      v_new = _concat(v_new)#tuplie -> tensor
      v_0 = torch.unsqueeze(torch.reshape(v_new, [-1]), 1)#更新新的一个v0以方便更新Gi(y)
      z_list.append(v_0)
    
    v_Q = eta*v_0 + torch.sum(torch.stack(z_list), dim=0)
    v_Q = torch.squeeze(v_Q)
    v_Q = self._handle_v_Q(v_Q,vector)
    return v_Q

  def _compute_vQ_softmax(self,input,target,eta,v_0,vector,hessian_q):
    z_list = []
    weight_list = []
    # loss = self.model._loss(input,target)
    # G_gradient = torch.autograd.grad(loss,self.model.parameters())
    v_0 = _concat(v_0)
    z_list.append(v_0)
    weight_list.append(v_0.norm())
    for i in range(hessian_q):
      v_0 =  self.infine_different(input,target,eta,v_0)#1399,48,3,3,3
      # else:
      #   v_new =  self.infine_different(input,target,eta,v_new)
      
      v_new = _concat(v_0)
      v_0 = torch.unsqueeze(torch.reshape(v_new, [-1]), 1)
      z_list.append(v_new)
      weight_list.append(v_new.norm())      
    k = hessian_q +1
    weight_param = torch.stack(weight_list)
    v_sum = torch.reshape(_concat(z_list),(k,-1))
    weight_softmax = torch.softmax(weight_param,dim=0)#4
    v_Q = torch.matmul(weight_softmax,v_sum)
    # v_Q = torch.sum(v_sum,dim=0)

    #v_Q = eta*v_0 + torch.sum(torch.stack(z_list), dim=0)
    v_Q = k*torch.squeeze(v_Q)
    v_Q = self._handle_v_Q(v_Q,vector)
    return v_Q
  
  def _compute_v_Q_softmax(self,input,target,eta,v_0,vector,hessian_q):
    z_list= []
    weight_list = []
    loss = self.model._loss(input,target)
    # loss.requires_grad_(True)
    Gw_gradient = torch.autograd.grad(loss, self.model.parameters())
    
    G_gradient = [(x-eta*y) for x, y in zip(self.model.parameters(), Gw_gradient)]
    
    
    G_gradient = _concat(G_gradient)
    v_0 = torch.unsqueeze(torch.reshape(v_0, [-1]), 1).detach()
    #Gw_gradient = torch.tensor(Gw_gradient) #solve q2 tuple->tensor
    #weight_list.append(v_0.sum())
    z_list.append(v_0)
    weight_list.append(v_0.norm())
    for _ in range(hessian_q):
      v_new =  [(x+eta*y) for x, y in zip( Gw_gradient,v_0)]
      
      Jacobian = torch.matmul(G_gradient,v_0)
      
      #x
      #q:grad can be implicitly created only for scalar outputs
      #因为auto出来的要是标量，而我这里要是tensor，所以要加上一个grad_outputs = torch.ones_like('tensor_variable')       
      v_new = torch.autograd.grad(Jacobian, self.model.parameters(),retain_graph=True)
      v_new = _concat(v_new)#tuplie -> tensor
      v_0 = torch.unsqueeze(torch.reshape(v_new, [-1]), 1)#更新新的一个v0以方便更新Gi(y)
      z_list.append(v_0)
      weight_list.append(v_0.norm())
    k = hessian_q +1
    weight_param = torch.stack(weight_list)
    v_sum = torch.reshape(_concat(z_list),(k,-1))
    weight_softmax = torch.softmax(weight_param,dim=0)#4
    
    
    v_Q = torch.matmul(weight_softmax,v_sum)
    # v_Q = torch.sum(v_sum,dim=0)

    #v_Q = eta*v_0 + torch.sum(torch.stack(z_list), dim=0)
    v_Q = k*torch.squeeze(v_Q)
    v_Q = self._handle_v_Q(v_Q,vector)
    return v_Q



  def _compute_v2_Q(self,input,target,eta,v_0,vector,hessian_q):
    z_list= []
    weight_list= []
    r = 1e-2
    R = r / _concat(vector).norm()
    #input ,target = input[1],target[1]
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.parameters())
    #v_1 = v_0
    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    Gww_gradient =  [(x-y).div_(2*r) for x, y in zip(grads_p, grads_n)]
    
    Gww_gradient = _concat(Gww_gradient)
    Gww_gradient = torch.unsqueeze(torch.reshape(Gww_gradient,[-1]), 1).detach()
    z_list.append(v_0)
    weight_list.append(v_0.norm())
    v_0 = torch.unsqueeze(torch.reshape(v_0,[-1]), 1).detach()
    #z_list.append(v_0)#不能放这里，放这里会导致出来的结构为（4,299578,1）
    #weight_list.append(v_0.sum())
    for _ in range(hessian_q):
      v_new = v_0 - eta * torch.mul(Gww_gradient,v_0)
      #v_1 = v_0 - loop * torch.matmul(Gww_gradient,v_0)
      v_0 = torch.unsqueeze(torch.reshape(v_new, [-1]), 1).detach()
      
      z_list.append(_concat(v_0))
      weight_list.append(v_0.norm())
    k = hessian_q +1
    weight_param = torch.stack(weight_list)
    v_sum = torch.reshape(_concat(z_list),(k,-1))
    weight_softmax = torch.softmax(weight_param,dim=0)#4
    
    v_Q = torch.matmul(weight_softmax,v_sum)
    #v_Q = torch.sum(v_Q,dim=-1)
    v_Q = k*torch.squeeze(v_Q)
    v_Q = self._handle_v_Q(v_Q,vector) 
    return v_Q#zhe




  def _compute_v3_Q(self,input,target,eta,v_0,vector,hessian_q):
      z_list= []
      weight_list = []
      r = 1e-2 
      loss = self.model._loss(input,target)
      v_x = torch.autograd.grad(loss,self.model.parameters())
      #v_0 = torch.unsqueeze(torch.reshape(v_0,[-1]), 1).detach()
      z_list.append(_concat(v_x))
      weight_list.append(_concat(v_x).sum())

      for _ in range(hessian_q):
        for p, v in zip(self.model.parameters(), v_x):
          p.data.add_(r, v)
        loss = self.model._loss(input,target)
        grads_ri = torch.autograd.grad(loss,self.model.parameters())

        for p, v in zip(self.model.parameters(), v_x):
          p.data.sub_(2*r, v)
        loss = self.model._loss(input, target)
        grads_le = torch.autograd.grad(loss, self.model.parameters())

        for p, v in zip(self.model.parameters(), vector):
          p.data.add_(r, v)

        Gww_gradient =  [(x-y).div_(2*r) for x, y in zip(grads_ri, grads_le)]

        v_x = [(x-eta*y) for x, y in zip(v_x, Gww_gradient)]
        v_new = _concat(v_x)
        #v_x = v_x -  Gww_gradient
        
        z_list.append(v_new)
        weight_list.append(v_new.sum())
      v_sum = torch.stack(z_list)#(4,299578)
      weight_param = torch.stack(weight_list)#(4,299578)

      weight_softmax = torch.nn.functional.softmax(weight_param)#4
      #v_Q = torch.matmul(v_sum.T,v_j)
      k = hessian_q+1
      v_Q = torch.matmul(k*weight_softmax,v_sum)
      #v_Q = torch.sum(v_sum, dim=0)
      v_Q = torch.squeeze(v_Q)
      v_Q = self._handle_v_Q(v_Q,vector)
      return v_Q#zhe
        
    
  def _handle_v_Q(self,v_Q,vector):
    z_list = ()
    k=0
    for i in range(len(vector)):
      g = k + vector[i].numel()
      v_new = torch.reshape(v_Q[k:g],vector[i].size())
      k = g
      z_list = z_list + (v_new,)
    v_new = z_list
    return v_new


  