from main import Main
from buffer.buffer import Buffer

import pandas as pd  

if __name__ =="__main__":
  buffer = Buffer()
  main = Main(21,5,5,buffer,2,10000,250)
  main.run()
  msg1 = buffer.agent1_msg
  msg2 = buffer.agent2_msg
  dict = {"msg1":msg1,"msg2":msg2}
  df = pd.DataFrame(dict) 
  df.to_csv('communication.csv')