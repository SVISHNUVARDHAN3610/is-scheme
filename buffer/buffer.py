class Buffer:
  def __init__(self):
    self.agent1_reward    = []
    self.agent2_reward    = []
    self.agent1_returns   = []
    self.agent2_returns   = []
    self.agent1_q_value   = []
    self.agent2_q_value   = []
    self.agent1_n_q_value = []
    self.agent2_n_q_value = []
    self.agent1_loss      = []
    self.agent2_loss      = []
    self.agent1_mgn_loss  = []
    self.agent2_mgn_loss  = []
    self.agent1_msg       = []
    self.agent2_msg       = []
    self.agent1_action    = []
    self.agent2_action    = []
    self.agent1_prob      = []
    self.agent2_prob      = []
    self.agent1_next_prob = []
    self.agent2_next_prob = []
    self.agent1_actor_loss= []
    self.agent2_actor_loss= []
    self.agent1_critic_loss=[]
    self.agent2_critic_loss=[]
    self.agent1_mean_reward=[]
    self.agent2_mean_reward=[]
    self.agent1_mean_critic=[]
    self.agent2_mean_critic=[]
    self.agent1_mean_loss = []
    self.agent2_mean_loss = []
    self.agent1_mean_mgn  = []
    self.agent2_mean_mgn  = []
    self.agent1_mean_actor= []
    self.agent2_mean_actor= [] 
    self.agent1_mean_value= []
    self.agent2_mean_value= []
    self.agent1_meannvalue= []
    self.agent2_meannvalue= []
    self.agent1_meanreturn= []
    self.agent2_meanreturn= []
    self.agent1_meanlog   = []
    self.agent2_meanlog   = []
    self.agent1_nextlog   = []
    self.agent2_nextlog   = []
    self.episodes         = []
    self.loss             = []
    self.mean_loss        = []