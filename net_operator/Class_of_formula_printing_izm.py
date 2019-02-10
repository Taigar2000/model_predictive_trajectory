import scipy, math
from sympy import *
import numpy as np
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 12:18:59 2019

@author: mac
"""

def popper(arr):
    if(len(arr)==0):
        return None
    if(len(arr)==1):
        return arr[0]
    if(len(arr)==2):
        return arr[0], arr[1]
    if(len(arr)==3):
        return arr[0], arr[1], arr[2]
    if(len(arr)==4):
        return arr[0], arr[1], arr[2], arr[3]
    if(len(arr)==5):
        return arr[0], arr[1], arr[2], arr[3], arr[4]
    if(len(arr)==6):
        return arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]

class Simplify:
    
    def __init__(self,input_nodes, output_nodes, psi, sym, nop):
        self.input_nodes = input_nodes#[0, 1, 2, 3, 4, 5]
        self.output_nodes = output_nodes#[0,18]
    
        #get_bp = [[(4,0),(0,1),(6,0)],[(5,0),(1,1),(7,0)],[(7,0),(6,0),(8,4)],[(8,0),(2,1),(9,0)],[(9,0),(10,5)]]
        #psi=[[(2,0),(3,1),(9,0)],[(9,0),(10,5)],[(4,0),(5,0),(11,1)],[(6,0),(11,0),(12,0)],
        #      [(0,0),(4,0),(10,5),(12,23),(13,1)],[(7,0),(0,19),(10,0),(14,1)],
        #      [(8,0),(0,0),(10,3),(5,0),(10,23),(15,1)],[(13,0),(14,0),(15,0),(16,0)],
        #      [(0,24),(16,0),(0,23),(17,1)],[(17,0),(0,25),(18,4)]]
        self.psi = psi #[[(0, 20), (2, 17), (5, 0)], [(5, 0), (2, 14), (3, 12), (18, 1)]]
        self.psil = None
        #self.q = q #{0: 0.04710989408847488, 1: 1.0, 2: 0.5}
        self.get_bp=psi
        #names_of_pars=["x0","x1","x2","x3","x4","x5"]
        self.names_of_pars = ['q0','q1','q2','x0','x1']#names_of_pars #["q0","q1","q2","x0","x1"]
        self.names_of_parsl = None
        self.sym = sym #[]
        #self.sym.append(symbols('x0'))
        #self.sym.append(symbols('x1'))
        self.nop = nop #[]
        #self.nop.append(q[0])
        #self.nop.append(q[1])
        #self.nop.append(q[2])
        self.qlen=len(self.nop)
        #nop.append(symbols('x0'))
        #nop.append(symbols('x1'))
        self.nop+=self.sym
        self.nopl=None
        self.noplen=len(self.nop)
    
    #constants
    L=1.5
    MAX_STEER = math.radians(45.0)
    N_IND_SEARCH = 10
    Kp = 1.0 
    dt = 0.125  # [s]
    dT = 10# [s]
    L = 1.5

    
    
    
    
    def addition(self, v):
        strok=str(v[0])
        for i in v[1:]:
            strok+='+'+str(i)
        try:
        	f=sympify(strok)

	        sm=self.sym
	        fm=lambda sm: f
	        #print(fm(self.sym))
	        gz=lambdify(self.sym,fm(self.sym),"sympy")
	        if(len(self.sym)==0):
	            return None
	        if(len(self.sym)==1):
	            return gz(self.sym[0])
	        if(len(self.sym)==2):
	            return gz(self.sym[0], self.sym[1])
	        if(len(self.sym)==3):
	            return gz(self.sym[0], self.sym[1], self.sym[2])
	        if(len(self.sym)==4):
	            return gz(self.sym[0], self.sym[1], self.sym[2], self.sym[3])
	        if(len(self.sym)==5):
	            return gz(self.sym[0], self.sym[1], self.sym[2], self.sym[3], self.sym[4])
	        if(len(self.sym)==6):
	            return gz(self.sym[0], self.sym[1], self.sym[2], self.sym[3], self.sym[4], self.sym[5])
        except:
        	return strok
        #gz=lambdify(self.sym,fm(self.sym),"sympy")
        #return f,strok
    
    def multiplication(self, v):
        strok=str(v[0])
        for i in v[1:]:
            strok+='*'+str(i)
        try:
        	f=sympify(strok)
	        
	        sm=self.sym
	        fm=lambda sm: f
	        gz=lambdify(self.sym,fm(self.sym),"sympy")
	        if(len(self.sym)==0):
	            return None
	        if(len(self.sym)==1):
	            return gz(self.sym[0])
	        if(len(self.sym)==2):
	            return gz(self.sym[0], self.sym[1])
	        if(len(self.sym)==3):
	            return gz(self.sym[0], self.sym[1], self.sym[2])
	        if(len(self.sym)==4):
	            return gz(self.sym[0], self.sym[1], self.sym[2], self.sym[3])
	        if(len(self.sym)==5):
	            return gz(self.sym[0], self.sym[1], self.sym[2], self.sym[3], self.sym[4])
	        if(len(self.sym)==6):
	            return gz(self.sym[0], self.sym[1], self.sym[2], self.sym[3], self.sym[4], self.sym[5])
	    except:
        	return strok
    
    binaries = [addition, multiplication]#, maximum, minimum, atan2, pi_2_pi]#, hypot, trapz]
    binariesn = ["addition", "multiplication", "maximum", "minimum", "atan2", "pi_2_pi"]#, hypot, trapz]
    
    L = 1.5
    ##unary##
    
    
    
    def relu(self, a):
        if(a>= self.qlen): return 'relu('+str(self.nopl[a])+')'
        if self.nopl[a] < 0: return 0
        else:
            return self.nopl[a]
    
    def identity(self, a):
        #print(a)
        #print(len(self.nopl))
        return str(self.nopl[a])
    
    def pow_two(self, a):
        return str(self.nopl[a])+ "**2"
        
    def negative(self, a):
        return "0-"+str(self.nopl[a])
    
    def irer(self, a):
        return "(" + str(self.nopl[a]) + ")/(np.fabs(" + str(self.nopl[a])")) * sqrt(fabs("+str(self.nopl[a])+"))"
        
    def reverse(self, x):
        return "1/("+str(self.nopl[x])+")"
        
    def exp(self, a):
        return "exp(" + str(self.nopl[a])+")"
    
    def expm1(self, x):
        return "expm1(" + str(self.nopl[x])+")"
        
    def exp2(self, x):
        return "2**(" + str(self.nopl[x]) + ")"
        
    def sign(self, x):
        #x = (self.nopl[x])/fabs(self.nopl[x])
        return "(" + str(self.nopl[x]) + ")/fabs(" + str(self.nopl[x]) + ")"
    
    def natlog(self, a):
        return "log(" + str(self.nopl[a]) + ")" 
    def log10(self, x):
        return "log10(" + str(self.nopl[x]) + ")"
    def log2(self, x):
        return "log2(" + str(self.nopl[x]) + ")"
    def log1p(self, x):
        return "log1p("+str(self.nopl[x])+")"
    
    def logic(self, a):
        #if a >= 0: return 1
        #else: return 0
        return 'logic('+str(self.nopl[a])+')'
        
    def cosinus(self, a):
        return "cos(" + str(self.nopl[a]) + ")"
    
    def sinus(self, a):
        return "sin(" + str(self.nopl[a]) + ")"
    
    def tan(self, x):
        return "tan(" + str(self.nopl[x]) + ")"
        
    def tanh(self, x):
        return "tanh(" + str(self.nopl[x]) + ")"
    
    def cubicroot(self, a):
        return "(" + str(self.nopl[a]) + ")**(1/3)"
    
    def atan(self, x):
        return "atan(" + str(self.nopl[x]) + ")"
    
    def cubic(self, a):
        return "(" + str(self.nopl[a]) + ")**3)"
        
    def absolute(self, a):
        return "fabs(" + str(self.nopl[a]) + ")"
    def sinc(self, x):
        return "sinc(" + str(self.nopl[x]) + ")"
    
    def inv(self, x):
        return "(" + str(self.nopl[x]) + ")**(-1)"
    
    #Constant functions
    def l(self, x):
        return str(L)
    
    def one(self, x):
        return 1.0
    
    
    unaries = [identity, negative, pow_two, sinus, cosinus, atan, exp, natlog, irer, cubic, reverse, cubicroot,
               expm1, exp2, log10, log2, log1p, absolute, tan, tanh, inv, l, one]#, sinc]
    unariesn = ["(", "(-", "pow_two(", "sinus(", "logic(", "cosinus(", "atan(", "exp(", "natlog(", "irer(", "cubic(", "reverse(", "cubicroot(",
               "expm1(", "exp2(", "sign(", "log10(", "log2(", "log1p(", "absolute(", "tan(", "tanh(", "relu(", "inv(", "L(", "one("]#, sinc]

    
        
    def simpli(self, text_form=1, simple_form=1):
        self.psil=self.psi.copy()
        self.nopl=self.nop.copy()
        self.names_of_parsl = self.names_of_pars
        #print(nop)
        #print(addition([0,'x0+x1','x1*3**(4/3)']))
        for i in self.psil:
            #print(i)
            if(simple_form):
                while(len(self.nopl)<=i[-1][0]):
                    self.nopl.append(None)
                arrn=[]
                for j in range(0,len(i)-1):
                    #print(str(i[j][1])+"   "+str(len(self.unaries)))
                    arrn.append(self.unaries[int(i[j][1])](self,int(i[j][0])))
                self.nopl[i[-1][0]]=self.binaries[min(i[-1][1],1)](self,arrn)
                
            if(text_form):
                strr=self.binariesn[i[-1][1]]+'['
                for j in range(0,len(i)-1):
                    #print(i[j][1])
                    strr+=self.unariesn[int(i[j][1])]+self.names_of_parsl[int(i[j][0])]+'), '
                strr=strr[:-2]+']'
                while(len(self.names_of_parsl)<=i[-1][0]):
                    self.names_of_parsl.append('-')
                self.names_of_parsl[i[-1][0]]=strr
        if(text_form):        
            for k in self.output_nodes:
                #strr=str(int(k))+"   "
                #print(k)
                #print(self.names_of_parsl[k])
                print({k:self.names_of_parsl[k]})
            print("")
        if(simple_form):
            for k in self.output_nodes:
                #strr=str(int(k))+"   "
                #print(k)
                #print(self.nopl[k])
                print({k:self.nopl[k]})
            
        #print(names_of_pars)
        #print(nop)
        
if __name__=='__main__':
    input_nodes = [0, 1, 2, 3, 4]
    output_nodes = [18]
    psi = [[(0, 20), (2, 17), (5, 0)], [(5, 0), (2, 14), (3, 12), (18, 1)]]
    #names_of_pars = ["q0","q1","q2","x0","x1"]
    names_of_inputs = ['x0','x1']
    names_of_params = ['q0','q1','q2']
    sym = symbols(names_of_inputs)
    #self.sym.append(symbols('x0'))
    #self.sym.append(symbols('x1'))
    q = {0: 0.04710989408847488, 1: 1.0, 2: 0.5}
    nop = [q[0],q[1],q[2]]
    
    s=Simplify(input_nodes,output_nodes,psi,sym,nop)
    s.simpli()
    
        