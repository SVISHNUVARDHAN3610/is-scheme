import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
from torch.autograd import variable as v
import sys
sys.path.append('./')
from buffer.ploting import Ploting
from Agents.Agent1 import Agent1
from Agents.Agent2 import Agent2
from Networks.Critic import Critic
from make_env import make_env
env = make_env("simple_reference")

class Main:
  def __init__(self,state_size,action_size,message_size,buffer,n_agents,n_games,steps):
    self.state_size = state_size
    self.action_size = action_size
    self.message_size = message_size
    self.n_agents    = n_agents
    self.n_games     = n_games
    self.steps       = steps
    self.gamma  = 0.99
    self.lr     = 0.0000007
    self.buffer = buffer
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.agent1 = Agent1(self.state_size,self.action_size,self.message_size,self.buffer)
    self.agent2 = Agent2(self.state_size,self.action_size,self.message_size,self.buffer)
    self.central= Critic(self.state_size,self.action_size,self.message_size).to(self.device)
    self.optim  = optim.Adam(self.central.parameters() , lr = self.lr)
    self.msg1   = []
    self.msg2   = []
  def choose_action(self,state,msg1,msg2):
    act  = []
    obs1 = torch.from_numpy(state[0]).float().to(self.device)
    obs2 = torch.from_numpy(state[1]).float().to(self.device)
    msg1 = torch.tensor(msg1,dtype = torch.float32).to(self.device)
    msg2 = torch.tensor(msg2,dtype = torch.float32).to(self.device)
    act1 = self.agent1.choose_action(obs1,msg2).cpu()
    act2 = self.agent2.choose_action(obs2,msg1).cpu()
    act.append(act1.detach().numpy())
    act.append(act2.detach().numpy())
    return act
  def next_value(self,next_state,action,r1,r2):
    obs1 = torch.from_numpy(next_state[0]).float().to(self.device)
    obs2 = torch.from_numpy(next_state[1]).float().to(self.device)
    r1   = torch.tensor([r1,0],dtype = torch.float32).to(self.device)
    r2   = torch.tensor([r2,0],dtype = torch.float32).to(self.device)
    nv1  = self.central(obs1,action[0],r1)
    nv2  = self.central(obs2,action[1],r2)
    return nv1,nv2
  def message(self,state,action,msg1,msg2):
    obs1 = torch.from_numpy(state[0]).float().to(self.device)
    obs2 = torch.from_numpy(state[1]).float().to(self.device)
    act1 = torch.tensor(action[0],dtype = torch.float32).to(self.device)
    act2 = torch.tensor(action[1],dtype = torch.float32).to(self.device)
    msg2 = torch.tensor(msg2,dtype = torch.float32).to(self.device)
    msg1 = self.agent1.message(obs1,act1,msg2)
    msg2 = self.agent2.message(obs2,act2,msg1)
    return msg1,msg2
  def update(self,state,next_state,reward,done,msg1,msg2):
    action     = self.choose_action(state,msg1,msg2)
    action     = torch.tensor(action,dtype = torch.float32).to(self.device)
    msg1,msg2  = self.message(state,action,msg1,msg2)
    agnv1,agnv2= self.next_value(next_state,action,reward[0],reward[1])
    agv1       = self.agent1.q_value(torch.from_numpy(state[0]).float().to(self.device),action[0],torch.tensor(reward[0],dtype = torch.float32))
    agv2       = self.agent2.q_value(torch.from_numpy(state[1]).float().to(self.device),action[1],torch.tensor(reward[1],dtype = torch.float32))
    agr1       = torch.tensor(self.agent1.ppo_iter(torch.tensor(reward[0],dtype = torch.float32).to(self.device),agv1,agnv1,done[0]))
    agr2       = torch.tensor(self.agent2.ppo_iter(torch.tensor(reward[1],dtype = torch.float32).to(self.device),agv2,agnv2,done[1]))
    returns    = torch.tensor(agr1 + agr2).to(self.device)
    loss       = returns - agv1-agv2-agnv1-agnv2
    loss       = torch.tensor(loss[0],requires_grad = True).to(self.device)
    self.buffer.loss.append(loss)
    torch.save(self.central.state_dict() , "memory/centeral.pth")
    self.agent1.learn(state[0],next_state[0],reward[0],done[0],agnv1,msg2)
    self.agent2.learn(state[1],next_state[1],reward[1],done[1],agnv2,msg1)
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()
  def mean(self,episode):
    if episode==0:
      self.buffer.mean_loss.append(sum(self.buffer.loss))
      self.buffer.agent1_mean_loss.append(sum(self.buffer.agent1_loss))
      self.buffer.agent1_mean_actor.append(sum(self.buffer.agent1_actor_loss))
      self.buffer.agent1_mean_critic.append(sum(self.buffer.agent1_critic_loss))
      self.buffer.agent1_mean_mgn.append(sum(self.buffer.agent1_mean_loss))    
      self.buffer.agent1_meanlog.append(sum(self.buffer.agent1_prob))
      self.buffer.agent1_nextlog.append(sum(self.buffer.agent1_next_prob))
      self.buffer.agent1_mean_value.append(sum(self.buffer.agent1_q_value))
      self.buffer.agent1_meanreturn.append(sum(self.buffer.agent1_returns))
      self.buffer.agent1_meannvalue.append(sum(self.buffer.agent1_n_q_value))
      self.buffer.agent1_mean_reward.append(sum(self.buffer.agent1_reward))
      self.buffer.agent2_mean_actor.append(sum(self.buffer.agent2_actor_loss))
      self.buffer.agent2_mean_critic.append(sum(self.buffer.agent2_critic_loss))
      self.buffer.agent2_mean_mgn.append(sum(self.buffer.agent2_mean_loss))    
      self.buffer.agent2_meanlog.append(sum(self.buffer.agent2_prob))
      self.buffer.agent2_nextlog.append(sum(self.buffer.agent2_next_prob))
      self.buffer.agent2_mean_value.append(sum(self.buffer.agent2_q_value))
      self.buffer.agent2_meanreturn.append(sum(self.buffer.agent2_returns))
      self.buffer.agent2_meannvalue.append(sum(self.buffer.agent2_n_q_value))
      self.buffer.agent2_mean_reward.append(sum(self.buffer.agent2_reward))
      self.buffer.agent2_mean_loss.append(sum(self.buffer.agent2_loss))
    else:
      self.buffer.mean_loss.append(sum(self.buffer.loss)/len(self.buffer.loss))
      self.buffer.agent1_mean_loss.append(torch.tensor(sum(self.buffer.agent1_loss)/len(self.buffer.agent1_loss)).cpu())
      self.buffer.agent2_mean_loss.append(torch.tensor(sum(self.buffer.agent2_loss)/len(self.buffer.agent2_loss)).cpu())
      self.buffer.agent1_mean_actor.append(torch.tensor(sum(self.buffer.agent1_actor_loss)/len(self.buffer.agent1_actor_loss)).cpu())
      self.buffer.agent1_mean_critic.append(torch.tensor(sum(self.buffer.agent1_critic_loss)/len(self.buffer.agent1_critic_loss)).cpu())
      self.buffer.agent1_mean_mgn.append(sum(self.buffer.agent1_mgn_loss)/len(self.buffer.agent1_mgn_loss))    
      self.buffer.agent1_meanlog.append(torch.tensor(sum(self.buffer.agent1_prob)/len(self.buffer.agent1_prob)).cpu())
      self.buffer.agent1_nextlog.append(torch.tensor(sum(self.buffer.agent1_next_prob)/len(self.buffer.agent1_next_prob)).cpu())
      self.buffer.agent1_mean_value.append(torch.tensor(sum(self.buffer.agent1_q_value)/len(self.buffer.agent1_q_value)).cpu())
      self.buffer.agent1_meanreturn.append(sum(self.buffer.agent1_returns)/len(self.buffer.agent1_returns))
      self.buffer.agent1_meannvalue.append(torch.tensor(sum(self.buffer.agent1_n_q_value)/len(self.buffer.agent1_n_q_value)).cpu())
      self.buffer.agent1_mean_reward.append(torch.tensor(sum(self.buffer.agent1_reward)/len(self.buffer.agent1_reward)).cpu())
      self.buffer.agent2_mean_actor.append(torch.tensor(sum(self.buffer.agent2_actor_loss)/len(self.buffer.agent2_actor_loss)).cpu())
      self.buffer.agent2_mean_critic.append(torch.tensor(sum(self.buffer.agent2_critic_loss)/len(self.buffer.agent2_critic_loss)).cpu())
      self.buffer.agent2_mean_mgn.append(sum(self.buffer.agent2_mgn_loss)/len(self.buffer.agent2_mgn_loss))    
      self.buffer.agent2_meanlog.append(torch.tensor(sum(self.buffer.agent2_prob)/len(self.buffer.agent2_prob)).cpu())
      self.buffer.agent2_nextlog.append(torch.tensor(sum(self.buffer.agent2_next_prob)/len(self.buffer.agent2_next_prob)).cpu())
      self.buffer.agent2_mean_value.append(torch.tensor(sum(self.buffer.agent2_q_value)/len(self.buffer.agent2_q_value)).cpu())
      self.buffer.agent2_meanreturn.append(sum(self.buffer.agent2_returns)/len(self.buffer.agent2_returns))
      self.buffer.agent2_meannvalue.append(torch.tensor(sum(self.buffer.agent2_n_q_value)/len(self.buffer.agent2_n_q_value)).cpu())
      self.buffer.agent2_mean_reward.append(torch.tensor(sum(self.buffer.agent2_reward)/len(self.buffer.agent2_reward)).cpu())
  def clear(self):
    self.buffer.loss = []
    self.buffer.agent2_actor_loss = []
    self.buffer.agent2_critic_loss = []
    self.buffer.agent2_prob = []
    self.buffer.agent2_next_prob= []
    self.buffer.agent2_q_value =[]
    self.buffer.agent2_returns  = []
    self.buffer.agent2_n_q_value = []
    self.buffer.agent2_reward = []
    self.buffer.agent1_actor_loss = []
    self.buffer.agent1_critic_loss = []
    self.buffer.agent1_prob = []
    self.buffer.agent1_next_prob= []
    self.buffer.agent1_q_value =[]
    self.buffer.agent1_returns  = []
    self.buffer.agent1_n_q_value = []
    self.buffer.agent1_reward = []
  def run(self):
    for i in range(self.n_games):
      state = env.reset()
      score = [0,0]
      done  = [False ,False]
      self.buffer.episodes.append(i)
      self.mean(i)
      self.clear()
      plts = Ploting(self.buffer)
      print("episode:",i,",","agent1_reward:",self.buffer.agent1_mean_reward[i],",","agent2_reward:",self.buffer.agent1_mean_reward[i])
      for step in range(self.steps):
        if i==0 & step==0:
          msg1 = torch.zeros(1)
          msg2 = torch.zeros(1)
          action  = self.choose_action(state,msg1,msg2)
          msg1,msg2 = self.message(state,action,msg1,msg2)
          self.msg1.append(msg1)
          self.msg2.append(msg2)
        else:
          msg1 = self.msg1[0]
          msg2 = self.msg2[0]
          action= self.choose_action(state,msg1,msg2)
          self.msg1 = []
          self.msg2 = []
          self.msg1.append(msg1)
          self.msg2.append(msg2)
        next_state,reward,done,info = env.step(action)
        self.buffer.agent1_reward.append(reward[0])
        self.buffer.agent2_reward.append(reward[1]) 
        self.update(state,next_state,reward,done,msg1,msg2)
        if done:
          state = next_state
          score += reward
        else:
          self.update(state,next_state,reward,done,msg1,msg2)
          state = next_state
          score += reward
          print("completed")