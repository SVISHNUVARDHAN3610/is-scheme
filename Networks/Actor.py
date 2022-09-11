import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
from torch.autograd import variable as v

class Actor(nn.Module):
  def __init__(self,state_size,action_size):
    super(Actor,self).__init__()
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
    self.fc10= nn.Linear(32,self.action_size).to(self.device)
    
  def forward(self,state,message):
    message   = torch.tensor([message.item(),0]).to(self.device)
    self.fs1 = nn.Linear(state.shape[0],32).to(self.device)
    self.fm1 = nn.Linear(message.shape[0],32).to(self.device)
    state    = torch.tensor(state).to(self.device)
    s1       = self.fs1(state).to(self.device)
    m1       = self.fm1(message)
    cat      = torch.cat([s1.view(32,-1),m1.view(32,-1)],0)
    cat      = torch.reshape(cat,(-1,)).to(self.device)
    self.fc2 = nn.Linear(cat.shape[0],64).to(self.device)
    x        = self.fc5(f.relu(self.fc4(f.relu(self.fc3(f.relu(self.fc2(cat)))))))
    x        = self.fc9(f.relu(self.fc8(f.relu(self.fc7(f.relu(self.fc6(f.relu(x))))))))
    x        = f.softmax(self.fc10(f.relu(x)))
    return x