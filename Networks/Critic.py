import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
from torch.autograd import variable as v

class Critic(nn.Module):
  def __init__(self,state_size,action_size,message_size):
    super(Critic,self).__init__()
    self.state_size  = state_size
    self.action_size = action_size
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.fc3 = nn.Linear(64,128).to(self.device)
    self.fc4 = nn.Linear(128,256).to(self.device)
    self.fc5 = nn.Linear(256,512).to(self.device)
    self.fc6 = nn.Linear(512,256).to(self.device)
    self.fc7 = nn.Linear(256,128).to(self.device)
    self.fc8 = nn.Linear(128,64).to(self.device)
    self.fc9 = nn.Linear(64,32).to(self.device)
    self.fc10= nn.Linear(32,1).to(self.device)
  def forward(self,state,action,reward):
    self.fs1 = nn.Linear(state.shape[0],32).to(self.device)
    self.fa1 = nn.Linear(action.shape[0],32).to(self.device)
    self.fr1 = nn.Linear(2,32).to(self.device)
    s1       = self.fs1(state)
    a1       = self.fa1(action)
    r1       = self.fr1(reward)
    cat      = torch.cat([s1.view(32,-1),r1.view(32,-1),a1.view(32,-1)],0)
    cat      = torch.reshape(cat,(-1,))
    self.fc2 = nn.Linear(cat.shape[0],64).to(self.device)
    x        = self.fc5(f.relu(self.fc4(f.relu(self.fc3(f.relu(self.fc2(cat)))))))
    x        = self.fc9(f.relu(self.fc8(f.relu(self.fc7(f.relu(self.fc6(f.relu(x))))))))
    x        = self.fc10(f.relu(x))
    return x