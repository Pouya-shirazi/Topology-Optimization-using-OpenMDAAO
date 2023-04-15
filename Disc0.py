# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 21:45:30 2023

@author: Pooya
"""
import numpy as np
import numpy.matlib
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import openmdao.api as om


DW=2
DH=1
nelx=80
nely=40
x_int=0.5
y_int=0.5
volfrac=0.4
ini_val= [0.38,0.04,0.06,0.04,0.7]


#Element stiffness matrix
def BasicKe(E,nu, a, b,h):
    k=np.array([-1/6/a/b*(nu*a**2-2*b**2-a**2), 1/8*nu+1/8, -1/12/a/b*(nu*a**2+4*b**2-a**2),3/8*nu-1/8, 1/12/a/b*(nu*a**2-2*b**2-a**2),-1/8*nu-1/8, 1/6/a/b*(nu*a**2+b**2-a**2), -3/8*nu+1/8])
    KE=E*h/(1-nu**2)*np.array(
    [ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
    [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
    [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
    [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
    [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
    [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
    [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
    [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])
    return KE

h=1 #thickness
E=1
nu=0.3

EW = DW / nelx 
EH = DH / nely  
x, y = np.meshgrid(EW * np.arange(nelx+1), EH * np.arange(nely+1))
LSgridx = x.T.reshape(-1)
LSgridy = y.T.reshape(-1)

# Component geometry initialization
x0=np.arange(x_int/2, DW, x_int)# x-coordinates of the centers of components
y0=np.arange(y_int/2, DH, y_int)# y-coordinates of the centers of components
xn=len(x0)# number of component groups in x direction
yn=len(y0)# number of component groups in y direction
x0=np.kron(x0,np.ones((1,2*yn)))
y0=np.matlib.repmat(np.kron(y0,np.ones((1,2))),1,xn)
N=x0.shape[1]# total number of components in the design domain
L=np.matlib.repmat(ini_val[0],1,N)# vector of the lf length of each component
t1=np.matlib.repmat(ini_val[1],1,N) # vector of the half width of component at point A
t2=np.matlib.repmat(ini_val[2],1,N) # vector of the half width of component at point B
t3=np.matlib.repmat(ini_val[3],1,N) # vector of the half width of component at point C
st=np.matlib.repmat([ini_val[4],-ini_val[4]],1,int(N/2)) # vector of the sine value of the inclined angle of each component  
variable=np.vstack((x0,y0,L,t1,t2,t3,st))

 #Limits of variable:[x0; y0; L; t1; t2; t3;st];
xmin=np.vstack((0, 0, 0.01, 0.01, 0.01, 0.03, -1.0))
xmin=np.matlib.repmat(xmin,N,1)
xmax=np.vstack((DW, DH, 2.0, 0.2, 0.2, 0.2, 1.0))
xmax=np.matlib.repmat(xmax,N,1)
low=xmin
upp=xmax
m=1; #number of constraint
Var_num=7 # number of design variablesfor each component
nn=Var_num*N
c=1000*np.ones((m,1))
d=np.zeros((m,1))
a0=1
a=np.zeros((m,1))

#Define loads and supports(Short beam)
fixeddofs=np.arange(0,(2*(nely+1)))
alldofs=np.arange(0,2*(nely+1)*(nelx+1))
freedofs=np.setdiff1d(alldofs,fixeddofs)
loaddof=2*(nely+1)*nelx+nely+1
F=csc_matrix(([-1], ([loaddof], [0])), shape=(2*(nely+1)*(nelx+1), 1))


##Preparation FE analysis
nodenrs=np.arange(0,(1+nelx)*(1+nely)).reshape(1+nely,1+nelx,order='F')
edofVec=((2*nodenrs[0:-1,0:-1])).reshape(nelx*nely,1,order='F')
edofMat=np.matlib.repmat(edofVec,1,8)+np.matlib.repmat(np.concatenate([np.array([0,1]),2*nely+np.array([2,3,4,5]),np.array([2,3])],axis=0),nelx*nely,1)
iK=np.kron(edofMat,np.ones((8,1))).T
jK=np.kron(edofMat,np.ones((1,8))).T
EleNodesID=(edofMat[:,(1,3,5,7)]-1)/2
iEner=EleNodesID.T
KE=BasicKe(E,nu, EW, EH,h) # stiffness matrix k**s is formed

