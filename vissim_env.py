import numpy as np
import math
import random
# from os import path
import copy
import collections as col
import os
import time


class VissimEnv:

    def __init__(self,raw_obs0):
        self.initial_run = True
        self.end=0
        self.max_comfort=1.8
        self.max_v=22
        self.pre_raw_obs=raw_obs0
        self.pre_acceleration=0
        self.time_step=0
        self.pre_LaneChanging=0

    def step(self, acceleration,LaneChanging,raw_obs):
       #print("Step")

        # Make an obsevation from a raw observation vector from Vissim
                
        # Reward setting Here #######################################
        # direction-dependent positive reward
        
        Vx = raw_obs[0]
        Vy = raw_obs[1]
        Dl = raw_obs[2]
        Dr = raw_obs[3]
        Vx2_diff = raw_obs[4]
        Dx2_diff = raw_obs[5]
        Vy2_diff = raw_obs[6]
        Dy2_diff = raw_obs[7]
        Vx1_diff = raw_obs[8]
        Dx1_diff = raw_obs[9]
        Vy1_diff = raw_obs[10]
        Dy1_diff = raw_obs[11]
        Vx3_diff = raw_obs[12]
        Dx3_diff = raw_obs[13]
        Vy3_diff = raw_obs[14]
        Dy3_diff = raw_obs[15]
        Vx6_diff = raw_obs[16]
        Dx6_diff = raw_obs[17]
        Vy6_diff = raw_obs[18]
        Dy6_diff = raw_obs[19]
        Vx4_diff = raw_obs[20]
        Dx4_diff = raw_obs[21]
        Vy4_diff = raw_obs[22]
        Dy4_diff = raw_obs[23]
        Vx5_diff = raw_obs[24]
        Dx5_diff = raw_obs[25]
        Vy5_diff = raw_obs[26]
        Dy5_diff = raw_obs[27]

        pre_Vx = self.pre_raw_obs[0]
        pre_Vy = self.pre_raw_obs[1]
        pre_Dl = self.pre_raw_obs[2]
        pre_Dr = self.pre_raw_obs[3]
        pre_Vx2_diff = self.pre_raw_obs[4]
        pre_Dx2_diff = self.pre_raw_obs[5]
        pre_Vy2_diff = self.pre_raw_obs[6]
        pre_Dy2_diff = self.pre_raw_obs[7]
        pre_Vx1_diff = self.pre_raw_obs[8]
        pre_Dx1_diff = self.pre_raw_obs[9]
        pre_Vy1_diff = self.pre_raw_obs[10]
        pre_Dy1_diff = self.pre_raw_obs[11]
        pre_Vx3_diff = self.pre_raw_obs[12]
        pre_Dx3_diff = self.pre_raw_obs[13]
        pre_Vy3_diff = self.pre_raw_obs[14]
        pre_Dy3_diff = self.pre_raw_obs[15]
        pre_Vx6_diff = self.pre_raw_obs[16]
        pre_Dx6_diff = self.pre_raw_obs[17]
        pre_Vy6_diff = self.pre_raw_obs[18]
        pre_Dy6_diff = self.pre_raw_obs[19]
        pre_Vx4_diff = self.pre_raw_obs[20]
        pre_Dx4_diff = self.pre_raw_obs[21]
        pre_Vy4_diff = self.pre_raw_obs[22]
        pre_Dy4_diff = self.pre_raw_obs[23]
        pre_Vx5_diff = self.pre_raw_obs[24]
        pre_Dx5_diff = self.pre_raw_obs[25]
        pre_Vy5_diff = self.pre_raw_obs[26]
        pre_Dy5_diff = self.pre_raw_obs[27]

        



        #reward=vel/22/abs(vel_diff + random.random())/abs(d-2*vel-4.25-random.random())      #/math.pow(max(0.01,abs(action-self.pre_action)),0.28)
        reward = Vx/22

        #car-following
        if Dx2_diff < Vx + 4.25:
            reward = -1
        if Dx2_diff > 5*Vx + 4.25:
            reward = -0.5

            
        if Vx < 1 and Vx2_diff < 0 and Dx2_diff > 15:
            if acceleration < 0:
                reward = -1

        if abs(acceleration - self.pre_acceleration) > 0.56:
            reward = -0.5

        #if LaneChanging == 1 or LaneChanging == 2:   #change a lane
            #if Dx5_diff < -Vx5_diff + 4.25:
                #reward = -1
            #if self.pre_LaneChanging == 1 or self.pre_LaneChanging == 2:
                #reward = -0.8

        self.pre_raw_obs=raw_obs
        
        self.pre_acceleration=acceleration

        self.time_step += 1

        #self.revise=np.array([self.observation, action])

        #self.observation=self.revise

        return reward

    def reset(self,raw_obs):        
        self.reset=1
        return self.make_observaton(raw_obs)

    def end(self):
         self.end=1
         #send end(1 for end and 0 for not)

    def get_obs(self):
        return self.observation

    def make_observaton(self, raw_obs):
        Observation = np.array([raw_obs[0]/22, raw_obs[1]/2, raw_obs[4]/11, raw_obs[5]/100, raw_obs[6]/2, raw_obs[7]/4, raw_obs[8]/11, raw_obs[9]/100, raw_obs[10]/2, raw_obs[11]/4, raw_obs[12]/11, raw_obs[13]/100, raw_obs[14]/2, raw_obs[15]/4, raw_obs[16]/11, raw_obs[17]/100, raw_obs[18]/2, raw_obs[19]/4, raw_obs[20]/11, raw_obs[21]/100, raw_obs[22]/2, raw_obs[23]/4, raw_obs[24]/11, raw_obs[25]/100, raw_obs[26]/2, raw_obs[27]/4])
        return Observation
