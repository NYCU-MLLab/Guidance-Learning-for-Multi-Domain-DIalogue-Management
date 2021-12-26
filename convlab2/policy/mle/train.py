import os
import torch
import logging
import torch.nn as nn

from convlab2.util.train_util import to_device

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLE_Trainer_Abstract():
    def __init__(self, manager, cfg):
        self._init_data(manager, cfg)
        self.policy = None
        self.policy_optim = None
        
    def _init_data(self, manager, cfg):
        self.data_train = manager.create_dataset('train', cfg['batchsz'])
        self.data_valid = manager.create_dataset('val', cfg['batchsz'])
        self.data_test = manager.create_dataset('test', cfg['batchsz'])
        self.save_dir = cfg['save_dir']
        self.print_per_batch = cfg['print_per_batch']
        self.save_per_epoch = cfg['save_per_epoch']
        self.multi_entropy_loss = nn.MultiLabelSoftMarginLoss()
        self.multi_entropy_loss1 = nn.MultiLabelSoftMarginLoss()
        self.multi_entropy_loss2 = nn.MultiLabelSoftMarginLoss()
        
    def policy_loop(self, data):
        s, target_a, target_bs, target_ua = to_device(data)
        a_weights,bs,ua = self.policy(s)
        
        loss_a = self.multi_entropy_loss(a_weights, target_a)
        loss_bs = self.multi_entropy_loss1(bs, target_bs)
        loss_ua = self.multi_entropy_loss2(ua, target_ua)
        return loss_a,loss_bs,loss_ua
        
    def imitating(self, epoch):
        """
        pretrain the policy by simple imitation learning (behavioral cloning)
        """
        self.policy.train()
        loss_act = 0.
        loss_bs = 0.
        loss_ua = 0.
        loss_tot = 0.
        for i, data in enumerate(self.data_train):
            self.policy_optim.zero_grad()
            los_a,los_bs,los_ua = self.policy_loop(data)
            loss_act+= los_a.item()
            loss_bs+= los_bs.item()
            loss_ua+= los_ua.item()
            loss_tot+= los_a.item()+los_bs.item()+los_ua.item()
            losses = 1*los_a+0.8*los_bs+0.6*los_ua
            losses.backward()
            self.policy_optim.step()
            
            if (i+1) % self.print_per_batch == 0:
                loss_tot /= self.print_per_batch
                loss_act /= self.print_per_batch
                loss_bs /= self.print_per_batch
                loss_ua /= self.print_per_batch
                logging.debug('<<dialog policy>> epoch {}, iter {}, loss_tot:{}, loss_act:{}, loss_bs:{}, loss_ua:{}'.format(epoch, i, loss_tot,loss_act,loss_bs,loss_ua))
                loss_act = 0.
                loss_bs = 0.
                loss_ua = 0.
                loss_tot = 0.
        
        if (epoch+1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch)
        self.policy.eval()
    
    def imit_test(self, epoch, best):
        """
        provide an unbiased evaluation of the policy fit on the training dataset
        """
        loss_act = 0.
        loss_bs = 0.
        loss_ua = 0.
        loss_tot = 0.
        for i, data in enumerate(self.data_valid):
            los_a,los_bs,los_ua = self.policy_loop(data)
            loss_act+= los_a.item()
            loss_bs+= los_bs.item()
            loss_ua+= los_ua.item()
            loss_tot+= los_a.item()+los_bs.item()+los_ua.item()

        loss_tot /= len(self.data_valid)
        loss_act /= len(self.data_valid)
        loss_bs /= len(self.data_valid)
        loss_ua /= len(self.data_valid)  

        logging.debug('<<dialog policy>>validation, epoch {}, iter {}, loss_tot:{}, loss_act:{}, loss_bs:{}, loss_ua:{}'.format(epoch, i, loss_tot,loss_act,loss_bs,loss_ua))
        if loss_act < best:
            logging.info('<<dialog policy>> best model saved')
            best = loss_act
            self.save(self.save_dir, 'best')
            
        loss_act = 0.
        loss_bs = 0.
        loss_ua = 0.
        loss_tot = 0.
        for i, data in enumerate(self.data_test):
            los_a,los_bs,los_ua = self.policy_loop(data)
            loss_act+= los_a.item()
            loss_bs+= los_bs.item()
            loss_ua+= los_ua.item()
            loss_tot+= los_a.item()+los_bs.item()+los_ua.item()

        loss_tot /= len(self.data_test)
        loss_act /= len(self.data_test)
        loss_bs /= len(self.data_test)
        loss_ua /= len(self.data_test)  

        logging.debug('<<dialog policy>>validation, epoch {}, iter {}, loss_tot:{}, loss_act:{}, loss_bs:{}, loss_ua:{}'.format(epoch, i, loss_tot,loss_act,loss_bs,loss_ua))
        return best

    def test(self):
        def f1(a, target):
            TP, FP, FN = 0, 0, 0
            real = target.nonzero().tolist()
            predict = a.nonzero().tolist()
            for item in real:
                if item in predict:
                    TP += 1
                else:
                    FN += 1
            for item in predict:
                if item not in real:
                    FP += 1
            return TP, FP, FN
    
        a_TP, a_FP, a_FN = 0, 0, 0
        for i, data in enumerate(self.data_test):
            s, target_a = to_device(data)
            a_weights = self.policy(s)
            a = a_weights.ge(0)
            TP, FP, FN = f1(a, target_a)
            a_TP += TP
            a_FP += FP
            a_FN += FN
            
        prec = a_TP / (a_TP + a_FP)
        rec = a_TP / (a_TP + a_FN)
        F1 = 2 * prec * rec / (prec + rec)
        print(a_TP, a_FP, a_FN, F1)

    def save(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)
        saved_mode = self.policy.state_dict()
        for key in list(saved_mode.keys()):
            if 'bs' in key or 'ua' in key:
                del saved_mode[key]
        
        torch.save(saved_mode, directory + '/' + str(epoch) + '_mle.pol.mdl')

        logging.info('<<dialog policy>> epoch {}: saved network to mdl'.format(epoch))

