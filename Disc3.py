# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 22:11:44 2023

@author: Pooya
"""

import numpy as np
import numpy.matlib
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import openmdao.api as om
from Disc0 import nely,nelx,freedofs,F,KE,iK,jK,EleNodesID




U=np.zeros((2*(nely+1)*(nelx+1),1))


class disc3(om.ImplicitComponent):
    
    def initialize(self):
        self.options.declare('denk', default=1.0)
        self.options.declare('sK', default=1.0)
        self.options.declare('U', default=1.0)

    def setup(self):
        
        self.add_input('H')
        
        self.add_output('Comp', val=1.0)
        

        
    def compute(self, inputs, outputs):
        self.options['denk']=np.sum(input['H'][EleNodesID.astype(int)]**2, axis=1) / 4
        self.options['sK']=np.expand_dims(KE.flatten(order='F'),axis=1)@(np.expand_dims(self.options['denk'].flatten(order='F'),axis=1).T)
        #FEA
        # Remove constrained dofs from matrix
        K = coo_matrix((self.options['sK'].flatten(order='F'),(iK.flatten(order='F'),jK.flatten(order='F'))),shape=(2*(nely+1)*(nelx+1),2*(nely+1)*(nelx+1))).tocsc()
        K = K[freedofs,:][:,freedofs]
        # Solve system 
        self.options['U'][freedofs,0]=spsolve(K,F[freedofs,0])
        
        outputs['Comp']=(F.T@self.options['U'])

        
    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')
        
        
        
        
        
        
        
