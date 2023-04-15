# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:45:52 2023

@author: Pooya
"""

import numpy as np
import numpy.matlib
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import openmdao.api as om
from Disc0 import EH,EW,N,Var_num,LSgridx,LSgridy,nely,nelx,EleNodesID

 
p=6
alpha=1e-3 # parameter alpha in the Heaviside function
epsilon=4*min(EW,EH) # regularization parameter epsilon in the Heaviside function
Phi=[None] * N
#Forming Phi_i for each component
def tPhi(xy,LSgridx,LSgridy,p):
    st=xy[6]
    ct=np.sqrt(abs(1-(st*st)))
    x1=ct*(LSgridx - xy[0])+st*(LSgridy - xy[1])
    y1=-st*(LSgridx - xy[0])+ct*(LSgridy -xy[1])
    bb=((xy[4]+xy[3]-2*xy[5])/(2*xy[2]**2))*(x1**2) + ((xy[4]-xy[3])/(2*xy[2]))*x1 + xy[5]
    tmpPhi=-(((x1)**p)/(xy[2]**p) + (((y1)**p)/(bb**p)) -1)
    return tmpPhi

#Heaviside function
def Heaviside(phi,alpha,nelx,nely,epsilon):
    phi=phi.flatten(order='F')
    num_all=np.arange(0,(nelx+1)*(nely+1))
    H=np.ones((nelx+1)*(nely+1))
    H=np.where(phi<-epsilon,alpha,H)
    H=np.where((phi>= -epsilon) & (phi<= epsilon),(3*(1-alpha)/4*(phi/epsilon-phi**3/(3*(epsilon)**3))+(1+alpha)/2),H)
    return H

#Forming Phi_i for each component
class disc1(om.ImplicitComponent):
    
    def setup(self):
        
        self.add_input('xy00',val=np.zeros(7*N))
        self.add_output('H', val=1.0)
       
        
    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')
        
        
    def compute(self, inputs, outputs):
        #Forming Phi^s
        for i in range(0,N):
            Phi[i]=tPhi(np.array(inputs['xy00'])[np.arange(Var_num*(i+1)-Var_num,Var_num*(i+1))],LSgridx,LSgridy,p)

        #Union of components
        tempPhi_max=Phi[0]

        for i in range(1,N):
            tempPhi_max=np.maximum(tempPhi_max,Phi[i])
            
        Phi_max=tempPhi_max.reshape((nely+1,nelx+1),order='F')

        
        outputs['H']=Heaviside(Phi_max,alpha,nelx,nely,epsilon)
        
        
        





