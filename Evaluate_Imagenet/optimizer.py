import math
import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required
import numpy as np

eta = -1e10

class Alpha:
    def __init__(self,alpha_matrix,weights_name=[],layer = 4):
        
        self.alpha_matrix = alpha_matrix
        self.weights_name = self.Node_info(weights_name)
        self.TF_alpha = np.ones([28,7],int)#标识每行alpha中每一个操作的使用情况
        self.pruning_weight = list()
        self.layer =  [layer//3, 2*layer//3]#标识
        self.alpha_g = self.init_alpha()
    #获取alpha与weight的对应关系
    def Node_info(self,weights_name):
        weights_index = list()
        for weight in weights_name:
            weights_index.append(weight.split('.'))
        return weights_index
           
    #初始化alpha字典
    def init_alpha(self):
        alpha_g = dict()
        #alpha 中的每一行
        for k,alpha in enumerate(self.alpha_matrix):
            alpha_gi = dict()
        #每一行中的7个操作
            if k < 28:
                for alpha_op in range(len(alpha[0])):
                #每一行的前三个操作是非参数操作
                    temp = list()
                    if alpha_op > 2:
                        for w_index,w_name in enumerate(self.weights_name):
                                if len(w_name) > 7 and ( int(w_name[5]) == alpha_op): 
                                    if k < 14  and (int(w_name[1]) not in self.layer) and int(w_name[3]) == k :
                                            #self.pruning_weight.append(w_index)
                                            temp.append(w_index)
                                    elif k >= 14  and (int(w_name[1]) in self.layer) and int(w_name[3]) == k-14 :
                                            #self.pruning_weight.append(w_index)
                                            temp.append(w_index)
                    alpha_gi[alpha_op] = temp
                alpha_g[k] = alpha_gi
        return alpha_g



class SGD(optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0,epochsize = 2,lambda_ = [0.001,0.01],C = -0.001 ,
                 weight_decay=0, nesterov=False,dampening = 0,alpha_lambda = 0.001,weights_name=[],layers = 4):
        defaults = dict(lr=lr, momentum=momentum, epochsize = epochsize,lambda_ = lambda_,C = C ,
                        weight_decay=weight_decay, nesterov=nesterov,dampening = dampening,alpha_lambda = alpha_lambda)
        
        self.weights_name = self.Node_info(weights_name)
        self.alpha = Alpha(params[1]['params'],weights_name= weights_name,layer = layers)
        self.layer =  [layers//3, 2*layers//3]
        self.pruning_weight = list()
        self.pruning_num = 0
        super(SGD, self).__init__(params, defaults)
        

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
    
    
    def Node_info(self,weights_name):
        weights_index = list()
        for weight in weights_name:
            weights_index.append(weight.split('.'))
        return weights_index    
    
    def Lambda(self,lam):
        self.defaults['lambda_'] = lam
        
    def CosLR(self, epoch, t = 5,n_t = 0.5,T = 50):
        lr = np.array(self.defaults['lr'])
        if epoch < t:
            lr = (0.9*epoch/t + 0.1)*lr
            lr[lr<0.001] = 0.001
            self.defaults['lr'] = lr 
            print(self.defaults['lr'])
        else:
            cosine = n_t * (1 + math.cos(math.pi * (epoch - t) / (T - t)))
            lr = lr*cosine
            lr[lr<0.001] = 0.001
            self.defaults['lr'] = lr 
            print(self.defaults['lr'])  
    def step1(self, epoch = 0,closure=None,opt_w = False,opt_alp = False):

        loss = None
        if closure is not None:
            loss = closure()
        weights_group = self.param_groups[0]
        alphas_group = self.param_groups[1]
        if opt_alp:
            #alphas_group['lr'][1] = self.CosLR(epoch,alphas_group['lr'][1],alphas_group['epochsize'])
            lr = self.defaults['lr'][1]
            #lambda_ = alphas_group['lambda_'][1]
            #C = alphas_group['C']
            #epochsize = alphas_group['epochsize']
            #epsilon = alphas_group['epsilon']
            self.pruning_num = 0
            alphas_group['amsgrad'] = False
            alphas_group['eps'] = 1e-8
            alphas_group['betas']=(0.5, 0.999)
            weight_decay = alphas_group['weight_decay'][1]

            for k, p in enumerate(alphas_group['params']):
                if p.grad is None or not p.requires_grad: #if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = alphas_group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = alphas_group['betas']

                state['step'] += 1

                if weight_decay != 0:
                    grad.add_(weight_decay, p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(alphas_group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(alphas_group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)
        if opt_w :
            #alphas_group['lr'][0] = self.CosLR(epoch,alphas_group['lr'][0],weights_group['epochsize'])
            weight_decay = weights_group['weight_decay'][0]
            momentum = weights_group['momentum']
            dampening = weights_group['dampening']
            nesterov = weights_group['nesterov']
            lr = self.defaults['lr'][0]

            for p in weights_group['params']:
                if p.requires_grad == False or p.grad==None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-lr, d_p)

    def step(self, epoch = 0,closure=None,opt_w = False,opt_alp = False):

        loss = None
        if closure is not None:
            loss = closure()
        weights_group = self.param_groups[0]
        alphas_group = self.param_groups[1]
        if opt_alp:
            #alphas_group['lr'][1] = self.CosLR(epoch,alphas_group['lr'][1],alphas_group['epochsize'])
            lr = self.defaults['lr'][1]
            #temp = [3.82066268e-07, 3.82066268e-07, 3.82066268e-07, 3.03599026e-01,3.63066876e-01, 1.51799513e-01, 1.81533438e-01]
            #temp = [1.0,1.0,1.0,1.0,1.0,1.0,1.0]
            temp = [1,1,1,794624,950272,397312,475136]
            lambda_ = torch.tensor(temp,dtype = torch.float32)
            lambda_ = 1.0 / lambda_
            #lambda_ = torch.tensor(temp,dtype = torch.float32)
            alpha_lambda = alphas_group['alpha_lambda']
            lambda_ = lambda_*(alpha_lambda)
            C = alphas_group['C']


            epochsize = alphas_group['epochsize']
            self.pruning_num = 0
            momentum = alphas_group['momentum']
            dampening = alphas_group['dampening']
            nesterov = alphas_group['nesterov']
            
            for k, p in enumerate(alphas_group['params']):
                if p.requires_grad == False or p.grad==None:
                    continue
                grad_f = p.grad.data
                if k > 27:
                    temp_grad = grad_f
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(temp_grad)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, temp_grad)
                    if nesterov:
                        temp_grad = temp_grad.add(momentum, buf)
                    else:
                        temp_grad = buf
                    s = -lr*temp_grad
                    p.data.add_(s,alpha=1)
                else:
                    norm = torch.norm(p.data-C, p=2, dim=1)
                    temp_grad = grad_f+lambda_*((p.data-C)/(norm + 1e-6))
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(temp_grad)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, temp_grad)
                    if nesterov:
                        temp_grad = temp_grad.add(momentum, buf)
                    else:
                        temp_grad = buf

                    s = -lr*temp_grad
                    if epoch <= epochsize:
                        p.data.add_(s,alpha=1)
                    else:
                        zero_num = torch.ones_like(p.data) 
                        z = p.data + s
                        
                        proj_idx = z*p.data < p.data**2
                        proj_idx_1 = z < C
                        proj_idx = proj_idx*proj_idx_1#交集
                        
                        p.data[proj_idx==True] = eta
                        p.data[proj_idx == False] = z[proj_idx == False]
                        
                        if torch.sum(zero_num[proj_idx==True]) > 0:
                            if sum(self.alpha.TF_alpha[k,:]) == 0:
                                self.pruning_weight.append(k)
                                p.requires_grad = False
                                
                            self.pruning_num += torch.sum(zero_num[proj_idx==True])
                            Temp = torch.nonzero(proj_idx[0]==True)#满组要求的具体操作
                            alpha_g = self.alpha.alpha_g[k]
                            for i in Temp:
                                if self.alpha.TF_alpha[k,i] == 1:
                                    self.alpha.TF_alpha[k,i] = 0
                                    for j in alpha_g[int(i)]:
                                        self.alpha.pruning_weight.append(j)
                                        weights_group['params'][j].requires_grad = False
                                        weights_group['params'][j].data = torch.zeros_like(weights_group['params'][j].data)            

        if opt_w :
            #alphas_group['lr'][0] = self.CosLR(epoch,alphas_group['lr'][0],weights_group['epochsize'])
            weight_decay = weights_group['weight_decay'][0]
            momentum = weights_group['momentum']
            dampening = weights_group['dampening']
            nesterov = weights_group['nesterov']
            lr = self.defaults['lr'][0]

            for p in weights_group['params']:
                if p.requires_grad == False or p.grad==None:
                    #print(p.requires_grad)
                    #print(p.grad)
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-lr, d_p)



