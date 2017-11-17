from vissim_env import VissimEnv
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
# from keras.engine.training import collect_trainable_weights
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit

from socket import *
from time import ctime
import struct
import socketserver
import h5py
from random import choice

OU = OU()  # Ornstein-Uhlenbeck Process


def playGame(train_indicator=1):  # 1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    #BUFFER_SIZE1 = 50000
    #BUFFER_SIZE2 = 5000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001  # Target Network HyperParameters
    LRA = 0.0001  # Learning rate for Actor
    LRC = 0.001  # Lerning rate for Critic

    action_dim = 2  # Acceleration/LaneChanging
    state_dim = 26  # of sensors input

    np.random.seed(1337)

    EXPLORE = 100000
    episode_count = 20000
    max_steps = 5299
    done = 0
    step = 0
    epsilon = 1

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff= ReplayBuffer(BUFFER_SIZE)
    #buff0 = ReplayBuffer(BUFFER_SIZE0)  # Create replay buffer
    #buff1 = ReplayBuffer(BUFFER_SIZE1)
    #buff2 = ReplayBuffer(BUFFER_SIZE2)
    # Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights("actormodel.h5")
        print("actor Weight load successfully")

        actor.target_model.load_weights("actor_target_model.h5")

    except:
        print("Cannot find the actor weight")

    try:
        critic.model.load_weights("criticmodel.h5")
        critic.target_model.load_weights("critic_target_model.h5")
        print("critic Weight load successfully")

    except:
        print("Cannot find the critic weight")

    HOST = '127.0.0.1'
    PORT = 5099
    BUFSIZ = 1024
    ADDR = (HOST, PORT)
    socketserver.TCPServer.allow_reuse_address = True
    tcpSerSock = socket(AF_INET, SOCK_STREAM)
    tcpSerSock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    tcpSerSock.bind(ADDR)
    tcpSerSock.listen(5)

    # while True:
    print ('waiting for connection...')
    tcpCliSock, addr = tcpSerSock.accept()
    print ('...connected from:', addr)

    #save Reward file
    with open("r_l_q_everyeposide.txt", "w") as f:
        print("Vissim Experiment Start.")
        for i in range(episode_count):

            print("Episode : " + str(i) + " Replay Buffer " + str(buff.num_experiences))

            data0 = tcpCliSock.recv(BUFSIZ)
            Vx0 = struct.unpack("30d", data0)[0]
            Vy0 = struct.unpack("30d", data0)[1]
            Dl0 = struct.unpack("30d", data0)[2]
            Dr0 = struct.unpack("30d", data0)[3]
            Vx2_diff0 = struct.unpack("30d", data0)[4]
            Dx2_diff0 = struct.unpack("30d", data0)[5]
            Vy2_diff0 = struct.unpack("30d", data0)[6]
            Dy2_diff0 = struct.unpack("30d", data0)[7]
            Vx1_diff0 = struct.unpack("30d", data0)[8]
            Dx1_diff0 = struct.unpack("30d", data0)[9]
            Vy1_diff0 = struct.unpack("30d", data0)[10]
            Dy1_diff0 = struct.unpack("30d", data0)[11]
            Vx3_diff0 = struct.unpack("30d", data0)[12]
            Dx3_diff0 = struct.unpack("30d", data0)[13]
            Vy3_diff0 = struct.unpack("30d", data0)[14]
            Dy3_diff0 = struct.unpack("30d", data0)[15]
            Vx6_diff0 = struct.unpack("30d", data0)[16]
            Dx6_diff0 = struct.unpack("30d", data0)[17]
            Vy6_diff0 = struct.unpack("30d", data0)[18]
            Dy6_diff0 = struct.unpack("30d", data0)[19]
            Vx4_diff0 = struct.unpack("30d", data0)[20]
            Dx4_diff0 = struct.unpack("30d", data0)[21]
            Vy4_diff0 = struct.unpack("30d", data0)[22]
            Dy4_diff0 = struct.unpack("30d", data0)[23]
            Vx5_diff0 = struct.unpack("30d", data0)[24]
            Dx5_diff0 = struct.unpack("30d", data0)[25]
            Vy5_diff0 = struct.unpack("30d", data0)[26]
            Dy5_diff0 = struct.unpack("30d", data0)[27]
            done0 = struct.unpack("30d", data0)[28]
            aux0 = struct.unpack("30d", data0)[29]
            raw_obs0=[Vx0,Vy0,Dl0,Dr0,Vx2_diff0,Dx2_diff0,Vy2_diff0,Dy2_diff0,Vx1_diff0,Dx1_diff0,Vy1_diff0,Dy1_diff0,Vx3_diff0,Dx3_diff0,Vy3_diff0,Dy3_diff0,Vx6_diff0,Dx6_diff0,Vy6_diff0,Dy6_diff0,Vx4_diff0,Dx4_diff0,Vy4_diff0,Dy4_diff0,Vx5_diff0,Dx5_diff0,Vy5_diff0,Dy5_diff0]
            print('raw_obs0=',raw_obs0)

            # Generate a Vissim environment
            env = VissimEnv(raw_obs0)

            s_t = env.make_observaton(raw_obs0)

            total_loss = 0
            total_reward_cf = 0
            total_reward_lc = 0
            total_q_value = 0

            for j in range(max_steps):
                loss = 0
                epsilon -= 1.0 / EXPLORE
                a_t = np.zeros([1,action_dim])
                noise_t = np.zeros([1, action_dim])

                a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))

                noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.60, 0.30)
                noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0.02, 1.00, 0.10)



                a_t[0][0] = a_t_original[0][0]+noise_t[0][0]
                a_t[0][1] = a_t_original[0][1]+noise_t[0][1]

                if a_t[0][1] > 0:
                    acceleration = a_t[0][1] * 3.5
                else:
                    acceleration = a_t[0][1] * 8

                if  0 <= a_t[0][0] and a_t[0][0] <= 0.1739523314093953:
                    LaneChanging = 1
                elif a_t[0][0] > 0.1739523314093953 and a_t[0][0] <= 1-0.1739523314093953:
                    LaneChanging = 0
                else:
                    LaneChanging = 2
                    
   
                #1 represent left lane changing
                #0 represent no lane changing
                #2 represent right lane changing
                
                ACTION=[LaneChanging, acceleration]
                
                print("acceleration=",acceleration)
                print("LaneChanging=",LaneChanging)
                #while True:

                tcpCliSock.send(str(ACTION).encode())

                data = tcpCliSock.recv(BUFSIZ)
                #print(data)

                Vx = struct.unpack("30d", data)[0]
                Vy = struct.unpack("30d", data)[1]
                Dl = struct.unpack("30d", data)[2]
                Dr = struct.unpack("30d", data)[3]
                Vx2_diff = struct.unpack("30d", data)[4]
                Dx2_diff = struct.unpack("30d", data)[5]
                Vy2_diff = struct.unpack("30d", data)[6]
                Dy2_diff = struct.unpack("30d", data)[7]
                Vx1_diff = struct.unpack("30d", data)[8]
                Dx1_diff = struct.unpack("30d", data)[9]
                Vy1_diff = struct.unpack("30d", data)[10]
                Dy1_diff = struct.unpack("30d", data)[11]
                Vx3_diff = struct.unpack("30d", data)[12]
                Dx3_diff = struct.unpack("30d", data)[13]
                Vy3_diff = struct.unpack("30d", data)[14]
                Dy3_diff = struct.unpack("30d", data)[15]
                Vx6_diff = struct.unpack("30d", data)[16]
                Dx6_diff = struct.unpack("30d", data)[17]
                Vy6_diff = struct.unpack("30d", data)[18]
                Dy6_diff = struct.unpack("30d", data)[19]
                Vx4_diff = struct.unpack("30d", data)[20]
                Dx4_diff = struct.unpack("30d", data)[21]
                Vy4_diff = struct.unpack("30d", data)[22]
                Dy4_diff = struct.unpack("30d", data)[23]
                Vx5_diff = struct.unpack("30d", data)[24]
                Dx5_diff = struct.unpack("30d", data)[25]
                Vy5_diff = struct.unpack("30d", data)[26]
                Dy5_diff = struct.unpack("30d", data)[27]
                done = struct.unpack("30d", data)[28]
                aux = struct.unpack("30d", data)[29]
                raw_obs=[Vx,Vy,Dl,Dr,Vx2_diff,Dx2_diff,Vy2_diff,Dy2_diff,Vx1_diff,Dx1_diff,Vy1_diff,Dy1_diff,Vx3_diff,Dx3_diff,Vy3_diff,Dy3_diff,Vx6_diff,Dx6_diff,Vy6_diff,Dy6_diff,Vx4_diff,Dx4_diff,Vy4_diff,Dy4_diff,Vx5_diff,Dx5_diff,Vy5_diff,Dy5_diff]


                print('vel=',Vx)
                print('vel_diff=',Vx2_diff)
                print('d=',Dx2_diff)
                print('done=',done)

                if raw_obs==[]:
                    print('No data')
                    break



                if LaneChanging == 1 or LaneChanging == 2:
                    r_t_lanechange = aux
                    if aux == -0.8:
                        r_t_follow = env.step(acceleration,raw_obs)
                    else:
                        r_t_follow = 0
                elif LaneChanging ==0:
                    r_t_follow = env.step(acceleration,raw_obs)
                    r_t_lanechange = 0

                if i == 0 and j == 0:
                    r_t_lanechange, r_t_follow = 0, 0

                print('r_t_follow=', r_t_follow,'r_t_lanechange=',r_t_lanechange)

                r_t = [r_t_follow, r_t_lanechange]

                s_t1 = env.make_observaton(raw_obs)

                q_value = critic.model.predict_on_batch([np.array(s_t).reshape(1,26), np.array(a_t_original).reshape(1,2)])
                target_q_value = critic.target_model.predict_on_batch([np.array(s_t).reshape(1,26), np.array(a_t_original).reshape(1,2)])
                #f.write("Episode" + str(i) + " " + "Step" + str(j) + " " + "Action=" + str(ACTION) + " " + "aIDM=" + str(aIDM) + "\n")
                error = abs(r_t + GAMMA*target_q_value - q_value)
                error = np.mean(error)
                # Add replay buffer
                buff.add(s_t, a_t[0], r_t, s_t1, done)

                #if error <= 1:
                    #buff0.add(s_t, a_t[0], r_t, s_t1, done)
                #elif error <= 5:
                    #buff1.add(s_t, a_t[0], r_t, s_t1, done)
                #else:
                    #buff2.add(s_t, a_t[0], r_t, s_t1, done)
                # Do the batch update
                #batch0 = buff0.getBatch(20)
                #batch1 = buff1.getBatch(10)
                #batch2 = buff2.getBatch(2)
                batch = buff.getBatch(BATCH_SIZE)

                #batch = []
                #if batch0 == []:
                    #pass
                #else:
                    #batch.append(batch0)
                #if batch1 == []:
                    #pass
                #else:
                    #batch.append(batch1)
                #if batch2 == []:
                    #pass
                #else:
                    #batch.append(batch2)
                states = np.asarray([e[0] for e in batch])
                actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                new_states = np.asarray([e[3] for e in batch])
                dones = np.asarray([e[4] for e in batch])
                y_t = np.asarray([e[2] for e in batch])

                #length = new_states.shape[0]

                target_q_values = critic.target_model.predict([new_states, actor.target_model.predict([new_states])])

                #q_values_of_batch = critic.model.predict([states, actions])

                #ave_q_values_of_batch = np.mean(q_values_of_batch)

                #ave_reward_of_batch = np.mean(rewards)

                for k in range(len(batch)):
                    if dones[k]:
                        y_t[k] = rewards[k]
                    else:
                        y_t[k] = rewards[k] + GAMMA*target_q_values[k]

                if (train_indicator):

                    loss += critic.model.train_on_batch([states, actions], y_t)
                    a_for_grad = actor.model.predict(states)
                    grads = critic.gradients(states, a_for_grad)
                    actor.train(states, grads)
                    actor.target_train()
                    critic.target_train()
                    
                    
                total_reward_cf += r_t_follow
                total_reward_lc += r_t_lanechange
                total_loss += loss
                total_q_value += q_value




                s_t = s_t1

                print("Episode", i, "Step", j, "Total Step", step, "acceleration=",acceleration,"LaneChanging=",LaneChanging, "Reward", r_t, "Loss", loss)

                step += 1

                if done==1:
                    break



            if np.mod(i, 5) == 0:
                if (train_indicator):
                    print("Now we save model")
                    actor.model.save_weights("actormodel.h5", overwrite=True)
                    with open("actormodel.json", "w") as outfile:
                        json.dump(actor.model.to_json(), outfile)

                    critic.model.save_weights("criticmodel.h5", overwrite=True)
                    with open("criticmodel.json", "w") as outfile:
                        json.dump(critic.model.to_json(), outfile)

                    critic.target_model.save_weights("critic_target_model.h5", overwrite=True)
                    with open("critic_target_model.json", "w") as outfile:
                        json.dump(critic.target_model.to_json(), outfile)

                    actor.model.save_weights("actor_target_model.h5", overwrite=True)
                    with open("actor_target_model.json", "w") as outfile:
                        json.dump(actor.target_model.to_json(), outfile)

            ave_loss = total_loss/(j+1)
            ave_q = total_q_value/(j+1)


            f.write("Episode" + str(i) + " " + "TotalReward_follow=" + str(total_reward_cf)+ " " + "TotalReward_lanechange=" + str(total_reward_lc) + " " + "AverageLoss=" + str(ave_loss) + " " + "AverageValue=" + str(ave_q)  + "\n")

        print("TOTAL REWARD @ " +str(j) +"/" +str(i) +"-th Episode  : Reward_follow " + str(total_reward_cf)+"Reward_follow :" + str(total_reward_lc))
        print("Total Step: " + str(step))
        print("")

        tcpCliSock.close()
        tcpSerSock.close()
        #env.end()  # This is for shutting down TORCS
        print("Finish.")

if __name__ == "__main__":
    playGame()
