import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
from torch.autograd import variable as v
import sys
sys.path.append('./')
from Networks.Actor import Actor
from Networks.Critic import Critic
from Networks.mgn import MGN

class Agent1:
  def __init__(self,state_size,action_size,message_size,buffer):
    self.state_size = state_size
    self.action_size  = action_size
    self.message_size = message_size
    self.gamma        = 0.99
    self.lamda        = 0.95
    self.lr1          = 0.0000009
    self.lr2          = 0.0000007
    self.buffer       = buffer
    self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.actor        = Actor(self.state_size,self.action_size).to(self.device)
    self.critic       = Critic(self.state_size,self.action_size,self.message_size).to(self.device)
    self.mgn          = MGN(self.state_size,self.action_size,self.message_size).to(self.device)
    self.actor_optim  = optim.Adam(self.actor.parameters() ,lr = self.lr1)
    self.critic_optim = optim.Adam(self.critic.parameters() ,lr = self.lr2)
    self.mgn_optim    = optim.Adam(self.mgn.parameters(),lr = self.lr2)
  def choose_action(self,observation,message):
    act               = self.actor(observation,message).to(self.device)
    return act
  def q_value(self,observation,action,reward):
    reward            = torch.tensor([reward.item(),0],dtype = torch.float32).to(self.device)
    value             = self.critic(observation,action,reward).to(self.device)
    return value
  def message(self,observation,action,message):
    message           = self.mgn(observation,action,message).to(self.device)
    return message
  def ppo_iter(self,reward,value,next_value,done):
    returns   = []
    gae       = 0
    for i in range(5):
      delta   = reward + self.gamma *(1-done)* next_value - value
      gae     = delta + self.gamma*self.lamda*gae*(1-done)
      returns.insert(0,gae + value+next_value)
    return returns
  def appending(self,loss,actor_loss,critic_loss,mgn,value,next_value,log_prob,returns,next_log_prob,action,msg1):
    self.buffer.agent1_mgn_loss.append(mgn)
    self.buffer.agent1_actor_loss.append(actor_loss)
    self.buffer.agent1_critic_loss.append(critic_loss)
    self.buffer.agent1_loss.append(loss)
    self.buffer.agent1_q_value.append(value)
    self.buffer.agent1_n_q_value.append(next_value)
    self.buffer.agent1_prob.append(log_prob.mean())
    self.buffer.agent1_returns.append(returns)
    self.buffer.agent1_msg.append(msg1.cpu())
    self.buffer.agent1_next_prob.append(next_log_prob.mean())
    self.buffer.agent1_action.append(action)
  def learn(self,state,next_state,reward,done,next_value,message):
    next_state        = torch.from_numpy(next_state).float().to(self.device)
    state             = torch.from_numpy(state).float().to(self.device)
    reward            = torch.tensor(reward , dtype = torch.float32).to(self.device)
    done              = torch.tensor(done   , dtype = torch.float32).to(self.device)
    action            = self.choose_action(state,message)
    next_action       = self.choose_action(next_state,message)
    message           = self.message(state,action,message)
    value             = self.q_value(state,action,reward)
    returns           = self.ppo_iter(reward.item(),value,next_value,done)
    returns           = torch.tensor(returns ,dtype = torch.float32).to(self.device)
    advantage         = returns - value -next_value
    log_prob          = torch.log(action).to(self.device)
    next_log          = torch.log(next_action).to(self.device)
    ratio             = (next_log - log_prob).exp()
    s1                = ratio * advantage
    s2                = torch.clamp(ratio,0.8,1.2)
    actor_loss        = torch.min(s1,s2).mean()
    critic_loss       = (returns - value)**2
    critic_loss       = torch.mean(critic_loss)
    actor_loss        = actor_loss
    loss              = actor_loss + 0.5*critic_loss
    mgn_loss          = message * log_prob * value
    mgn_loss          = torch.mean(mgn_loss)
    mgn_loss          = torch.tensor(mgn_loss,requires_grad=True).to(self.device)
    torch.save(self.actor.state_dict(),"memory/Agent1/actor.pth")
    torch.save(self.critic.state_dict(),"memory/Agent1/critic.pth")
    torch.save(self.mgn.state_dict(),"memory/Agent1/mgn.pth")
    self.appending(loss,actor_loss,critic_loss,mgn_loss,value,next_value,log_prob,returns,next_log,action,message)
    self.actor_optim.zero_grad()
    self.critic_optim.zero_grad()
    loss.backward()
    self.actor_optim.step()
    self.critic_optim.step()
    self.mgn_optim.zero_grad()
    mgn_loss.backward()
    self.mgn_optim.step()