import math
import numpy as np

class Memory():
    def __init__(self, params):
        self.obs = []
        self.acts = []
        self.advs = []
        self.est_rs = []

        self.batch_size = params.batch_size
        self.shuffle = params.shuffle

    def clear(self):
        self.obs = []
        self.acts = []
        self.advs = []
        self.est_rs = []
        

    def one_iteration(self):
        
        self.obs  = np.array( self.obs  )
        self.acts = np.array( self.acts )
        # print(self.acts)
        # print(self.acts.shape)
        self.advs = np.array( self.advs )
        self.est_rs = np.array( self.est_rs )

        total_len = len( self.obs )
        # batch_num = math.ceil( total_len / self.batch_size )
        
        idx = np.arange(total_len)
        if  self.shuffle:
            idx = np.random.permutation( total_len )

        pos = 0
        while pos < total_len:
            yield self.obs[idx[ pos : pos + self.batch_size ], : ], \
                self.acts[idx[ pos : pos + self.batch_size ], : ], \
                self.advs[idx[ pos : pos + self.batch_size ]], \
                self.est_rs[idx[ pos : pos + self.batch_size ]]
            pos += self.batch_size
