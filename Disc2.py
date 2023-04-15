# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 17:51:52 2023

@author: Pooya
"""

import numpy as np
import numpy.matlib
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import openmdao.api as om
from Disc0 import EH,EW,EleNodesID,DW,DH,volfrac,edofMat,KE,iEner,Var_num,N
from Disc3 import U


class disc2(om.ImplicitComponent):
    
    def setup(self):
        self.add_input('H')
        self.add_output('dvdh')
        
    
    def compute(self,inputs,outputs):

        den=np.sum(inputs['H'][EleNodesID.astype(int)], axis=1) / 4
        A1=np.sum(den)*EW*EH
        fval=A1/(DW*DH)-volfrac
        _, _, ix = np.unique(EleNodesID, return_index=True, return_inverse=True)
        C = np.bincount(ix)
        outputs['dvdh'] = EW*EH/4 * C

