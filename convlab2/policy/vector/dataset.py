import torch.utils.data as data

class ActDataset(data.Dataset):
    def __init__(self, s_s, a_s, b_s, u_a):
        self.s_s = s_s
        self.a_s = a_s
        self.b_s = b_s
        self.u_a = u_a
        self.num_total = len(s_s)
    
    def __getitem__(self, index):
        s = self.s_s[index]
        a = self.a_s[index]
        b_s = self.b_s[index]
        u_a = self.u_a[index]
        return s, a, b_s, u_a
    
    def __len__(self):
        return self.num_total
    
class ActStateDataset(data.Dataset):
    def __init__(self, s_s, a_s, next_s):
        self.s_s = s_s
        self.a_s = a_s
        self.next_s = next_s
        self.num_total = len(s_s)
    
    def __getitem__(self, index):
        s = self.s_s[index]
        a = self.a_s[index]
        next_s = self.next_s[index]
        return s, a, next_s
    
    def __len__(self):
        return self.num_total