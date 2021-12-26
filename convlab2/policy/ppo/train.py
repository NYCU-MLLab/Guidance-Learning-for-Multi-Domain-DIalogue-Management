# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:14:07 2019
@author: truthless
"""
import sys, os
import numpy as np
import torch
from torch import multiprocessing as mp
from convlab2.dialog_agent.agent import PipelineAgent
from convlab2.dialog_agent.env import Environment
from convlab2.nlu.svm.multiwoz import SVMNLU
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.policy.ppo import PPO
from convlab2.policy.rlmodule import Memory, Transition
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from argparse import ArgumentParser
import logging


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    mp = mp.get_context('spawn')
except RuntimeError:
    pass

def sampler(pid, queue, evt, env, policy, batchsz):
    """
    This is a sampler function, and it will be called by multiprocess.Process to sample data from environment by multiple
    processes.
    :param pid: process id
    :param queue: multiprocessing.Queue, to collect sampled data
    :param evt: multiprocessing.Event, to keep the process alive
    :param env: environment instance
    :param policy: policy network, to generate action from current policy
    :param batchsz: total sampled items
    :return:
    """
    buff = Memory()

    # we need to sample batchsz of (state, action, next_state, reward, mask)
    # each trajectory contains `trajectory_len` num of items, so we only need to sample
    # `batchsz//trajectory_len` num of trajectory totally
    # the final sampled number may be larger than batchsz.

    sampled_num = 0
    sampled_traj_num = 0
    traj_len = 50
    real_traj_len = 0

    #counter =0
    while sampled_num < batchsz:
        # for each trajectory, we reset the env and get initial state
        s = env.reset()
        counter = 0
        warn = 0
        for t in range(traj_len):
            

            # [s_dim] => [a_dim]
            s_vec = torch.Tensor(policy.vector.state_vectorize(s)[0])
            cur_dom = policy.vector.state_vectorize1(s)
            a = policy.predict(s)

            # interact with env
            next_s, r, done = env.step(a)

            # a flag indicates ending or not
            mask = 0 if done else 1

            # get reward compared to demostrations
            next_s_vec = torch.Tensor(policy.vector.state_vectorize(next_s)[0])
            next_dom = policy.vector.state_vectorize1(next_s)
            if torch.all(torch.eq(next_s_vec,s_vec)) and r<0:
                print(cur_dom +' '+ next_dom)
                counter +=1
                print(counter)
                if counter == 3:
                    warn=1
                    buff.delete()
                    buff.delete()
                    s_vec1 = s_vec
                    a1 = a
                    next_s_vec1 = next_s_vec
                    r1 = r
                    print(r1)
                    mask1 = mask
                    #buff.push(s_vec1.numpy(),policy.vector.action_vectorize(a1),r1,next_s_vec1.numpy(),mask1)
                    human = RulePolicy()
                    a = human.predict(s)
                    next_s, r, done = env.step(a)

                    mask = 0 if done else 1
                    next_s_vec = torch.Tensor(policy.vector.state_vectorize(next_s)[0])
                    if torch.all(torch.eq(next_s_vec1,next_s_vec)):
                        print('repeated state')
                    counter = 0  
                    
            # save to queue
            if warn == 0:
                human = RulePolicy()
                ax = human.predict(s)
                buff.push(s_vec.numpy(), policy.vector.action_vectorize(a), r, next_s_vec.numpy(),mask,policy.vector.action_vectorize(a),policy.vector.action_vectorize(ax))
            else:
                buff.push(s_vec1.numpy(),policy.vector.action_vectorize(a),r1,next_s_vec.numpy(),mask,policy.vector.action_vectorize(a1),policy.vector.action_vectorize(a))
                warn = 0

            # update per step
            s = next_s
            real_traj_len = t

            if done:
                break

        # this is end of one trajectory
        sampled_num += real_traj_len
        sampled_traj_num += 1
        # t indicates the valid trajectory length

    # this is end of sampling all batchsz of items.
    # when sampling is over, push all buff data into queue
    queue.put([pid, buff])
    evt.wait()


def sample(env, policy, batchsz, process_num):
    """
    Given batchsz number of task, the batchsz will be splited equally to each processes
    and when processes return, it merge all data and return
	:param env:
	:param policy:
    :param batchsz:
	:param process_num:
    :return: batch
    """

    # batchsz will be splitted into each process,
    # final batchsz maybe larger than batchsz parameters
    process_batchsz = np.ceil(batchsz / process_num).astype(np.int32)
    # buffer to save all data
    queue = mp.Queue()

    # start processes for pid in range(1, processnum)
    # if processnum = 1, this part will be ignored.
    # when save tensor in Queue, the process should keep alive till Queue.get(),
    # please refer to : https://discuss.pytorch.org/t/using-torch-tensor-over-multiprocessing-queue-process-fails/2847
    # however still some problem on CUDA tensors on multiprocessing queue,
    # please refer to : https://discuss.pytorch.org/t/cuda-tensors-on-multiprocessing-queue/28626
    # so just transform tensors into numpy, then put them into queue.
    evt = mp.Event()
    processes = []
    for i in range(process_num):
        process_args = (i, queue, evt, env, policy, process_batchsz)
        processes.append(mp.Process(target=sampler, args=process_args))
    for p in processes:
        # set the process as daemon, and it will be killed once the main process is stoped.
        p.daemon = True
        p.start()

    # we need to get the first Memory object and then merge others Memory use its append function.
    pid0, buff0 = queue.get()
    for _ in range(1, process_num):
        pid, buff_ = queue.get()
        buff0.append(buff_)  # merge current Memory into buff0
    evt.set()

    # now buff saves all the sampled data
    buff = buff0

    return buff.get_batch()


def update(env, policy, batchsz, epoch, process_num):
    # sample data asynchronously
    batch = sample(env, policy, batchsz, process_num)

    s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
    a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
    r = torch.from_numpy(np.stack(batch.reward)).to(device=DEVICE)
    mask = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
    act_weight = torch.Tensor(np.stack(batch.act_cor1)).to(device=DEVICE)
    act_target = torch.Tensor(np.stack(batch.act_cor2)).to(device=DEVICE)
    batchsz_real = s.size(0)

    policy.update(epoch, batchsz_real, s, a, r, mask,act_weight,act_target)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str, default='', help="path of model to load")
    parser.add_argument("--batchsz", type=int, default=1024, help="batch size of trajactory sampling")
    parser.add_argument("--epoch", type=int, default=20, help="number of epochs to train")
    parser.add_argument("--process_num", type=int, default=8, help="number of processes of trajactory sampling")
    args = parser.parse_args()

    # simple rule DST
    dst_sys = RuleDST()

    policy_sys = PPO(True)
    policy_sys.load(args.load_path)

    # not use dst
    dst_usr = None
    # rule policy
    policy_usr = RulePolicy(character='usr')
    # assemble
    simulator = PipelineAgent(None, None, policy_usr, None, 'user')

    evaluator = MultiWozEvaluator()
    env = Environment(None, simulator, None, dst_sys, evaluator)

    for i in range(args.epoch):
        update(env, policy_sys, args.batchsz, i, args.process_num)