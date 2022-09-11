import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
from torch.autograd import variable as v

class  MGN(nn.Module):
  def __init__(self,state_size,action_size,message_size):
    super(MGN,self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.message_size = message_size
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.fo3 = nn.Linear(64,128).to(self.device)
    self.fo4 = nn.Linear(128,256).to(self.device)
    self.fo5 = nn.Linear(256,512).to(self.device)
    self.fo6 = nn.Linear(512,256).to(self.device)
    self.fo7 = nn.Linear(256,128).to(self.device)
    self.fo8 = nn.Linear(128,64).to(self.device)
    self.fo9 = nn.Linear(64,32).to(self.device)
    self.f10 = nn.Linear(32,21).to(self.device)
  def action_predictor(self,observation):
    self.fa1 = nn.Linear(observation.shape[0],32).to(self.device)
    self.fa2 = nn.Linear(32,64).to(self.device)
    self.fa3 = nn.Linear(64,128).to(self.device)
    self.fa4 = nn.Linear(128,256).to(self.device)
    self.fa5 = nn.Linear(256,512).to(self.device)
    self.fa6 = nn.Linear(512,256).to(self.device)
    self.fa7 = nn.Linear(256,128).to(self.device)
    self.fa8 = nn.Linear(128,64).to(self.device)
    self.fa9 = nn.Linear(64,32).to(self.device)
    self.fa10= nn.Linear(32,self.action_size).to(self.device)
    obs = self.fa1(observation)
    x   = self.fa5(f.relu(self.fa4(f.relu(self.fa3(f.relu(self.fa2(f.relu(obs))))))))
    x   = self.fa9(f.relu(self.fa8(f.relu(self.fa7(f.relu(self.fa6(f.relu(x))))))))
    x   = f.softmax(self.fa10(f.relu(x)))
    return x
  def observation_predictor(self,prsent_action,other_action,observation):
    self.fo1 = nn.Linear(observation.shape[0],32).to(self.device)
    self.fa1 = nn.Linear(prsent_action.shape[0],32).to(self.device)
    self.fa2 = nn.Linear(other_action.shape[0],32).to(self.device)
    o        = self.fo1(observation)
    at       = self.fa1(prsent_action)
    at1      = self.fa2(other_action)
    cat      = torch.cat([o.view(32,-1),at.view(32,-1),at1.view(32,-1)],0)
    cat      = torch.reshape(cat,(-1,))
    self.fo2 = nn.Linear(cat.shape[0],64).to(self.device)
    x        = self.fo5(f.relu(self.fo4(f.relu(self.fo3(f.relu(self.fo2(cat)))))))
    x        = self.fo9(f.relu(self.fo8(f.relu(self.fo7(f.relu(self.fo6(x)))))))
    x        = f.relu(self.f10(f.relu(x)))
    return x
  def policy(self,message,observation):
    self.fm1  = nn.Linear(message.shape[0],32).to(self.device)
    self.fs1  = nn.Linear(observation.shape[0],32).to(self.device)
    m         = self.fm1(message)
    o         = self.fs1(observation)
    cat       = torch.cat([m.view(32,-1),o.view(32,-1)],0)
    cat       = torch.reshape(cat,(-1,))
    self.fc1  = nn.Linear(cat.shape[0],64).to(self.device)
    self.fc2  = nn.Linear(64,128).to(self.device)
    self.fc3  = nn.Linear(128,256).to(self.device)
    self.fc4  = nn.Linear(256,512).to(self.device)
    self.fc5  = nn.Linear(512,256).to(self.device)
    self.fc6  = nn.Linear(256,128).to(self.device)
    self.fc7  = nn.Linear(128,64).to(self.device)
    self.fc8  = nn.Linear(64,32).to(self.device)
    self.fc9  = nn.Linear(32,self.action_size).to(self.device)
    x         = self.fc5(f.relu(self.fc4(f.relu(self.fc3(f.relu(self.fc2(f.relu(self.fc1(cat)))))))))
    x         = self.fc8(f.relu(self.fc7(f.relu(self.fc6(f.relu(x))))))
    x         = f.softmax(self.fc9(f.relu(x)))
    return x
  def IGTM(self,message,observation,action):
    other_action   = self.action_predictor(observation)
    observation_t1 = self.observation_predictor(action,other_action,observation)
    policy         = self.policy(message,observation_t1)
    return message,observation_t1,policy
  def attention(self,message,observation,policy):
    trajectory    = torch.cat([observation,policy],0)
    trajectory    = torch.reshape(trajectory , (-1,))
    self.wv       = nn.Linear(trajectory.shape[0],128).to(self.device)
    self.wk       = nn.Linear(trajectory.shape[0],128).to(self.device)
    self.wq       = nn.Linear(message.shape[0],128).to(self.device)
    v             = self.wv(trajectory)
    k             = self.wk(trajectory)
    q             = self.wq(message)
    x1            = torch.matmul(q,k)
    x2            = f.softmax(x1)
    x1            = torch.tensor([x1.item(),0])
    x2            = torch.tensor([x2.item(),0])
    x             = torch.matmul(x1,x2)
    return x #message
  def forward(self,observation,action,message):
    message                     = torch.tensor([message.item(),0]).to(self.device)
    msg,observationt2,policy    =  self.IGTM(message,observation,action)
    msg                         =  self.attention(msg,observationt2,policy)
    return msg