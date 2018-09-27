import math
import numpy as np

class Memory():
    def __init__(self, params):
        self.obs = []
        self.acts = []
        self.advantages = []
        self.est_rs = []

        self.batch_size = params.batch_size
        self.shuffle = params.shuffle

    def one_iteration():
        
        self.obs = np

        total_len = len( self.obs )
        batch_num = math.ceil( total_len / self.batch_size )

        idx = np.arange( total_len )
        idx = np.shuffle( idx )

        for i in range( batch_num ):

