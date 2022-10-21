#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gurobipy as grb
from gurobipy import GRB
import numpy as np
from numpy import random
import random
import statistics
import math
import matplotlib.pyplot as plt


# In[2]:


class Graph(object):
    def __init__(self,size,weight):
        self.adjMatrix = []
        for i in range(size):
            self.adjMatrix.append([0 for i in range(size)])
        self.size = size
        self.weight = weight 
    def addEdge(self, v1, v2):
        if v1 == v2:
            print("Same vertex %d and %d" % (v1, v2))
        self.adjMatrix[v1][v2] = self.weight
        self.adjMatrix[v2][v1] = self.weight
        
        
    def removeEdge(self, v1, v2):
        if self.adjMatrix[v1][v2] == 0:
            print("No edge between %d and %d" % (v1, v2))
            return
        self.adjMatrix[v1][v2] = 0
        self.adjMatrix[v2][v1] = 0
    def containsEdge(self, v1, v2):
        return True if self.adjMatrix[v1][v2] > 0 else False
    def __len__(self):
        return self.size
            
    def printGraph(self):
        return self.adjMatrix
#for a tree woth two children make sure that size entered is of form  2**i -1    
    def BalancedTreeGraph(self):
        iterator = int((self.size-3)/2)
        for i in range(iterator+1):
            self.addEdge(i,(2*i)+1) 
            self.addEdge(i,(2*i)+2)
        
    
    def UnbalancedGraph(self):     
        self.BalancedTreeGraph()
        self.removeEdge(1,3)
        self.addEdge(2,3)
        
      
    def erdosRenyi(self):
        for i in range(self.size):
            for j in range(i+1,self.size,1):
                self.addEdge(i,j)
        
    
    def findEdgeSet(self):
        EdgeSet = []
        upperTriMatrix = []
        for i in range(self.size):
            upperTriMatrix.append([0 for i in range(self.size)])
        for i in range(self.size):
            for j in range(i,self.size,1):
                upperTriMatrix[i][j] = self.adjMatrix[i][j]
        for i in range(self.size):
            for j in range(self.size):
                if upperTriMatrix[i][j] > 0:
                    EdgeSet.append((i,j))
        return EdgeSet

    def findNodeEdgeMatrix(self):
        EdgeSet = self.findEdgeSet()
        A_node_edge_matrix = np.zeros(shape=(self.size,len(EdgeSet)),dtype = int)
        #for i in range(self.size): #for each node of graph g
            #A_node_edge_matrix.append([0 for j in range(len(EdgeSet))])
        for i in range(len(EdgeSet)):# nodes x edges matrix
            A_node_edge_matrix[EdgeSet[i][0]][i] = 1 
            A_node_edge_matrix[EdgeSet[i][1]][i] = -1 
    
        return A_node_edge_matrix
    
    

#given a mean and variance, generate demand. 

    def generate_demand(self,d_mean,d_variance):
        demandNodeSet = {}
        x = random_fixed_sum(d_mean,d_variance,self.size)
        for i in range(self.size):
            demandNodeSet[i] = x[i]
        #x = np.random.lognormal(d_mean,d_variance,self.size).tolist()
        #for i in range(self.size):
            #demandNodeSet[i] = min(int(x[i]),d_mean*3)
            #if demandNodeSet[i] == 0:
                #demandNodeSet[i] = 1
        return demandNodeSet

    def generate_supply(self,s_mean,s_variance): #make sure s_mean is larger than d_mean
        supplyNodeSet = {}
        x = random_fixed_sum(s_mean,s_variance,self.size)
        for i in range(self.size):
            supplyNodeSet[i] = x[i]
        return supplyNodeSet


    def generate_edge_capacity(self,f_mean,f_variance):
        EdgeSet = self.findEdgeSet()
        capacity_edges = {}
        x = random_fixed_sum(f_mean,f_variance,len(EdgeSet))
        #x = np.random.lognormal(f_mean,f_variance,len(EdgeSet)).tolist()
        for i in range(len(EdgeSet)):
            #capacity_edges[EdgeSet[i]] = min(int(x[i]),f_mean*2)
            capacity_edges[EdgeSet[i]] = x[i]
            #if capacity_edges[EdgeSet[i]] ==0:
                #capacity_edges[EdgeSet[i]] = 1
        return capacity_edges

'''
def CentralizedSol(A_sup,supply_cap,supplyNodeSet,total_demand):


    social_optimal_model = grb.model("centralised")
    supply_capacity = {}
    for i in supplierNodeSet:
        supply_capacity[i] = uniform_supply_capacity[i]
    
    s = social_optimal_model.addVars(supplierNodeSet, lb = 0, ub = supply_capacity,vtype= GRB.CONTINUOUS)
    social_optimal_model.addConstr(lhs = grb.quicksum(s[i] for i in range(1,num_nodes)),sense = grb.GRB.EQUAL,rhs = total_demand)
    social_optimal_model.addConstr(lhs = grb.quicksum(s[i] for i in range(1,num_nodes)),sense = grb.GRB.EQUAL,rhs = total_demand)


    social_optimal_model.setObjective(grb.quicksum(s[i]**alpha),grb.GRB.MINIMIZE)
'''    
def VoCAffine(demandNodeSet,supplyNodeSet,capacity_edges,A_node_edge_matrix,EdgeSet):
    network_imbalance = []
    sum_supply_cap = sum(supplyNodeSet.values())
    total_demand = sum(demandNodeSet.values())
    for i in supplyNodeSet:
        index_connected_edges = np.nonzero(A_node_edge_matrix[i])[0].tolist()
        connected_edges = [EdgeSet[j] for j in index_connected_edges]
        connected_capacity = 0
        for k in connected_edges:
            connected_capacity += capacity_edges[k]
        try:
            element = (demandNodeSet[i] + connected_capacity)/(sum_supply_cap - total_demand - supplyNodeSet[i])
        except:
            element = (demandNodeSet[i] + connected_capacity)
        network_imbalance.append(element)
    network_imbalance = max(network_imbalance)
    try :
        supply_imbalance = max(supplyNodeSet.values())/(sum_supply_cap - total_demand - max(supplyNodeSet.values()))
    except:
        supply_imbalance = max(supplyNodeSet.values())
    return max(network_imbalance,supply_imbalance)

def VoCLinear(demandNodeSet,supplyNodeSet,capacity_edges,A_node_edge_matrix,EdgeSet):
    network_imbalance = []
    sum_supply_cap = sum(supplyNodeSet.values())
    total_demand = sum(demandNodeSet.values())
    for i in supplyNodeSet:
        index_connected_edges = np.nonzero(A_node_edge_matrix[i])[0].tolist()
        connected_edges = [EdgeSet[j] for j in index_connected_edges]
        connected_capacity = 0
        for k in connected_edges:
            connected_capacity += capacity_edges[k]
        try:
            element = (demandNodeSet[i] + connected_capacity)/(total_demand - 2*(demandNodeSet[i] + connected_capacity))
            #print('element = ' + str(element))
        except:
            element = (demandNodeSet[i] + connected_capacity)
            print("error:element divide by zero")
        network_imbalance.append(element)
    
    network_imbalance = max(network_imbalance)
    print(network_imbalance)
    try :
        supply_imbalance = max(supplyNodeSet.values())/(total_demand - 2*max(supplyNodeSet.values()))
        print('supply_imbalance = ' + str(supply_imbalance))
    except:
        supply_imbalance = max(supplyNodeSet.values())
        print("error:supply imbalance divide by zero")
    #print('voc_linear = ' + str(max(network_imbalance,supply_imbalance)))
    return max(network_imbalance,supply_imbalance)

def random_fixed_sum(mean,var,num):
    total_sum = mean*num
    rand_set = []
    for x in range(num):
        x = min(np.random.lognormal(np.log(1),var,size=None),1.5*(mean))
        rand_set.append(x) 
        total_sum -= x
    return rand_set        
#total_demand = sum(uniform_demand)
#generalized inverse of matrix A is 
#Gen_inverse = numpy.linalg.pinv(A,rcond=1e-15, hermitian=False)
#Then solution of Ax = y is x= Gy# 


'''


for i in range(g.size):
    supply_demand_diff = -supplyNodeSet.get(i,0) + demandNodeSet.get(i,0)


gen_inverse = np.linalg.pinv(A_node_edge_matrix,rcond=1e-15, hermitian=False)
flow_edges = gen_inverse*(numpy.supply_demand_diff.transpose())

'''


# In[3]:


###Define a graph ######
graph = Graph(200,1)
graph.erdosRenyi()
EdgeSet = graph.findEdgeSet()

A_node_edge_matrix = graph.findNodeEdgeMatrix()
(24,6,2)
(30,6,5)
(24,8,2)
(30,8,5)
############################################################################################################
#################Generate supply demand and edge capacities ######
DemandNodeSet = graph.generate_demand(np.log(2),np.log(1))
SupplyNodeSet = graph.generate_supply(np.log(3),np.log(6))

capacity_edges = graph.generate_edge_capacity(np.log(20),np.log(1))

affine_voc= VoCAffine(DemandNodeSet,SupplyNodeSet,capacity_edges,A_node_edge_matrix,EdgeSet)
linear_voc= VoCLinear(DemandNodeSet,SupplyNodeSet,capacity_edges,A_node_edge_matrix,EdgeSet)

print(affine_voc,linear_voc)


# In[4]:


#####Plot with supply variance 
fig1, ax_a = plt.subplots(1,3, figsize=(15, 3), sharex=True,sharey= True)
fig2, ax_l = plt.subplots(1,3, figsize=(15, 3), sharex=True,sharey= True)
scale_list = [1, 1.5, 2]
fig_xlabel = ['a','b','c']
network_size = 100
sample = 30
x_axis = [i*.5 for i in range(20)]
keys = [(12,5,20),(20,5,20),(12,8,20)] # \mu_s, \mu_d,\mu_e,(30,8,4)
linestyle_tuple = [(0, (1, 10)),(0, (5, 10)),(0, (1, 1)),(0, (3, 10, 1, 10))]
###Define a graph ######
graph = Graph(network_size,1)
graph.erdosRenyi()
EdgeSet = graph.findEdgeSet()
A_node_edge_matrix = graph.findNodeEdgeMatrix()

############################################################################################################
for x in range(len(scale_list)):
    scale_factor = scale_list[x]
    affine_voc = {}
    linear_voc = {}
    for key in keys:
        affine_voc[key] = []
        linear_voc[key] = []
        DemandNodeSet = graph.generate_demand(key[1]*scale_factor,0)
        capacity_edges = graph.generate_edge_capacity(key[2],0)
        for i in range(20):
            total_affine = 0 
            total_linear = 0
            for num_sample in range(sample):
                SupplyNodeSet = graph.generate_supply(key[0]*scale_factor,i*.5)
                total_affine += VoCAffine(DemandNodeSet,SupplyNodeSet,capacity_edges,A_node_edge_matrix,EdgeSet)
                total_linear += VoCLinear(DemandNodeSet,SupplyNodeSet,capacity_edges,A_node_edge_matrix,EdgeSet)

            affine_voc[key].append(total_affine/sample)
            linear_voc[key].append(total_linear/sample)
     
    i = 0 
    for key in keys:
        ax_a[x].plot(x_axis,affine_voc[key],linestyle=linestyle_tuple[i],color='k',linewidth=3,label='$\mu_s$ = %.2f'%key[0]+' $\mu_d$ = %1.0f'%key[1]+' $\mu_e$ = %1.0f'%key[2])
        i += 1
        
    ax_a[x].set_ylabel('VoC')
    ax_a[x].set_xlabel('$\sigma_s^2$ \n \n (' + str(fig_xlabel[x]) + ') \n $\gamma$ = %.2f'%scale_factor) 
    
    i= 0 
    for key in keys:
        ax_l[x].plot(x_axis,linear_voc[key],linestyle=linestyle_tuple[i],color='k',linewidth=3,label='$\mu_s$ = %.2f'%key[0]+' $\mu_d$ = %1.0f'%key[1]+' $\mu_e$ = %1.0f'%key[2])
        i += 1
    
    ax_l[x].set_ylabel('VoC')
    ax_l[x].set_xlabel('$\sigma_s^2$ \n \n (' + str(fig_xlabel[x]) + ') \n $\gamma$ = %.2f'%scale_factor) 
    
    
    
 ######################################scale = 1#####################################3
ax_a[0].legend(loc='best')
ax_l[0].legend(loc='best')
for ax in ax_a.flat:
    ax.label_outer()

for ax in ax_l.flat:
    ax.label_outer()
fig1.savefig('Voc_affine_with_supply_var.jpg',dpi=300,bbox_inches = 'tight')
fig2.savefig('Voc_linear_with_supply_var.jpg',dpi=300,bbox_inches = 'tight')
plt.show()

           


# In[ ]:





# In[5]:


#####Plot with demand variance 
fig3, ax_a = plt.subplots(1,3, figsize=(15, 3), sharex=True,sharey= True)
fig4, ax_l = plt.subplots(1,3, figsize=(15, 3), sharex=True,sharey= True)
scale_list = [1, 1.5, 2]
fig_xlabel = ['a','b','c']
network_size = 100
sample = 30
x_axis = [i*.5 for i in range(20)]
keys = [(12,5,20),(20,5,20),(12,8,20)] # \mu_s, \mu_d,\mu_e,(30,8,4)
linestyle_tuple = [(0, (1, 10)),(0, (5, 10)),(0, (1, 1)),(0, (3, 10, 1, 10))]
###Define a graph ######
graph = Graph(network_size,1)
graph.erdosRenyi()
EdgeSet = graph.findEdgeSet()
A_node_edge_matrix = graph.findNodeEdgeMatrix()

############################################################################################################
for x in range(len(scale_list)):
    scale_factor = scale_list[x]
    affine_voc = {}
    linear_voc = {}
    for key in keys:
        affine_voc[key] = []
        linear_voc[key] = []
        SupplyNodeSet = graph.generate_supply(key[0]*scale_factor,0)
        capacity_edges = graph.generate_edge_capacity(key[2],0)
        for i in range(20):
            total_affine = 0 
            total_linear = 0
            for num_sample in range(sample):
                DemandNodeSet = graph.generate_demand(key[1]*scale_factor,i*.5)
                total_affine += VoCAffine(DemandNodeSet,SupplyNodeSet,capacity_edges,A_node_edge_matrix,EdgeSet)
                total_linear += VoCLinear(DemandNodeSet,SupplyNodeSet,capacity_edges,A_node_edge_matrix,EdgeSet)

            affine_voc[key].append(total_affine/sample)
            linear_voc[key].append(total_linear/sample)
     
    i = 0 
    for key in keys:
        ax_a[x].plot(x_axis,affine_voc[key],linestyle=linestyle_tuple[i],color='k',linewidth=3,label='$\mu_s$ = %.2f'%key[0]+' $\mu_d$ = %1.0f'%key[1]+' $\mu_e$ = %1.0f'%key[2])
        i += 1
        
    ax_a[x].set_ylabel('VoC')
    ax_a[x].set_xlabel('$\sigma_d^2$ \n \n (' + str(fig_xlabel[x]) + ') \n $\gamma$ = %.2f'%scale_factor) 
    
    i= 0 
    for key in keys:
        ax_l[x].plot(x_axis,linear_voc[key],linestyle=linestyle_tuple[i],color='k',linewidth=3,label='$\mu_s$ = %.2f'%key[0]+' $\mu_d$ = %1.0f'%key[1]+' $\mu_e$ = %1.0f'%key[2])
        i += 1
    
    ax_l[x].set_ylabel('VoC')
    ax_l[x].set_xlabel('$\sigma_d^2$ \n \n (' + str(fig_xlabel[x]) + ') \n $\gamma$ = %.2f'%scale_factor) 
    
    
    
 ######################################scale = 1#####################################3
ax_a[0].legend(loc='best')
ax_l[0].legend(loc='best')
for ax in ax_a.flat:
    ax.label_outer()

for ax in ax_l.flat:
    ax.label_outer()
fig3.savefig('Voc_affine_with_demand_var.jpg',dpi=300,bbox_inches = 'tight')
fig4.savefig('Voc_linear_with_demand_var.jpg',dpi=300,bbox_inches = 'tight')
plt.show()


# In[ ]:





# In[6]:


############with edge capacity variance and mean############
fig5, ax_a = plt.subplots(1,3, figsize=(15, 3), sharex=True,sharey= True)
fig6, ax_l = plt.subplots(1,3, figsize=(15, 3), sharex=True,sharey= True)
scale_list = [1, 1.5, 2]
fig_xlabel = ['a','b','c']
network_size = 100
sample = 30
x_axis = [i*.5 for i in range(20)]
keys = [(50,20,4),(50,20,8),(50,20,12)] # \mu_s, \mu_d,\mu_e,(30,8,4)
linestyle_tuple = [(0, (1, 10)),(0, (5, 10)),(0, (1, 1)),(0, (3, 10, 1, 10))]
###Define a graph ######
graph = Graph(network_size,1)
graph.erdosRenyi()
EdgeSet = graph.findEdgeSet()
A_node_edge_matrix = graph.findNodeEdgeMatrix()

############################################################################################################
for x in range(len(scale_list)):
    scale_factor = scale_list[x]
    affine_voc = {}
    linear_voc = {}
    for key in keys:
        affine_voc[key] = []
        linear_voc[key] = []
        SupplyNodeSet = graph.generate_supply(key[0]*scale_factor,0)
        DemandNodeSet = graph.generate_demand(key[1]*scale_factor,0)
        for i in range(20):
            total_affine = 0 
            total_linear = 0
            for num_sample in range(sample):
                capacity_edges = graph.generate_edge_capacity(key[2],i*.5)
                total_affine += VoCAffine(DemandNodeSet,SupplyNodeSet,capacity_edges,A_node_edge_matrix,EdgeSet)
                total_linear += VoCLinear(DemandNodeSet,SupplyNodeSet,capacity_edges,A_node_edge_matrix,EdgeSet)

            affine_voc[key].append(total_affine/sample)
            linear_voc[key].append(total_linear/sample)
     
    i = 0 
    for key in keys:
        ax_a[x].plot(x_axis,affine_voc[key],linestyle=linestyle_tuple[i],color='k',linewidth=3,label='$\mu_s$ = %.2f'%key[0]+' $\mu_d$ = %1.0f'%key[1]+' $\mu_e$ = %1.0f'%key[2])
        i += 1
        
    ax_a[x].set_ylabel('VoC')
    ax_a[x].set_xlabel('$\sigma_e^2$ \n \n (' + str(fig_xlabel[x]) + ') \n $\gamma$ = %.2f'%scale_factor) 
    
    i= 0 
    for key in keys:
        ax_l[x].plot(x_axis,linear_voc[key],linestyle=linestyle_tuple[i],color='k',linewidth=3,label='$\mu_s$ = %.2f'%key[0]+' $\mu_d$ = %1.0f'%key[1]+' $\mu_e$ = %1.0f'%key[2])
        i += 1
    
    ax_l[x].set_ylabel('VoC')
    ax_l[x].set_xlabel('$\sigma_e^2$ \n \n (' + str(fig_xlabel[x]) + ') \n $\gamma$ = %.2f'%scale_factor) 
    
ax_a[0].legend(loc='best')
ax_l[0].legend(loc='best')
for ax in ax_a.flat:
    ax.label_outer()

for ax in ax_l.flat:
    ax.label_outer()
fig5.savefig('Voc_affine_with_edge_cap_var.jpg',dpi=300,bbox_inches = 'tight')
fig6.savefig('Voc_linear_with_edge_cap_var.jpg',dpi=300,bbox_inches = 'tight')
plt.show()


# In[7]:



######Plot with network size####
fig7, ax_a = plt.subplots(1,3, figsize=(15, 3), sharex=True,sharey= True)
fig8, ax_l = plt.subplots(1,3, figsize=(15, 3), sharex=True,sharey= True)
scale_list = [1, 1.5, 2]
fig_xlabel = ['a','b','c']
network_length = list(range(30,300,10))
######################################################################

for x in range(len(scale_list)):
    affine_voc = []
    linear_voc = []
    scale_factor = scale_list[x]
    for i in range(30,300,10):
        graph = Graph(i,1)
        graph.erdosRenyi()
        EdgeSet = graph.findEdgeSet()
        A_node_edge_matrix = graph.findNodeEdgeMatrix()
    
        #################Generate supply demand and edge capacities ######
        DemandNodeSet = graph.generate_demand(15*scale_factor,0.5)
        capacity_edges = graph.generate_edge_capacity(10,0.5)
        SupplyNodeSet = graph.generate_supply(50*scale_factor,0.5)
        
        
        affine_voc.append(VoCAffine(DemandNodeSet,SupplyNodeSet,capacity_edges,A_node_edge_matrix,EdgeSet))
        linear_voc.append(VoCLinear(DemandNodeSet,SupplyNodeSet,capacity_edges,A_node_edge_matrix,EdgeSet))

    ax_a[x].plot(network_length,affine_voc,linestyle=':',color='k',linewidth=3,label='Affine Supply')  
    ax_l[x].plot(network_length,linear_voc,linestyle='--',color='k',linewidth=3,label='Linear Supply')
    ax_a[x].set_ylabel('VoC')
    ax_l[x].set_ylabel('VoC')
    ax_a[x].set_xlabel('network size \n \n (' + str(fig_xlabel[x]) + ') \n $\gamma$ = %.2f'%scale_factor)
    ax_l[x].set_xlabel('network size \n \n (' + str(fig_xlabel[x]) + ') \n $\gamma$ = %.2f'%scale_factor)


ax_a[0].legend(loc='best')
ax_l[0].legend(loc='best')
for ax in ax_a.flat:
    ax.label_outer()

for ax in ax_l.flat:
    ax.label_outer()
fig7.savefig('Voc_affine_with_network_length.jpg',dpi=300,bbox_inches = 'tight')
fig8.savefig('Voc_linear_with_network_length.jpg',dpi=300,bbox_inches = 'tight')
plt.show()


    


# In[ ]:





# # Degrees of Centralization Code 

# In[8]:


graph = Graph(7,1)
graph.erdosRenyi()
EdgeSet = graph.findEdgeSet()

A_node_edge_matrix = graph.findNodeEdgeMatrix()
d_mean = 10
factor = 1.5
############################################################################################################
    
    
#################Generate supply demand and edge capacities ######
DemandNodeSet = graph.generate_demand(d_mean,0)
capacity_edges = graph.generate_edge_capacity(20,0)
    
    
#Fix Total supply#
total_supply = int(graph.size*d_mean*factor) 

list_decentralization_degree = [i/graph.size for i in range(1,graph.size)]
affine_voc = []
linear_voc = []
for num_supplier in range(1,graph.size): #num_supplier is degree of decentralization
    supplyNodeList = random.sample(range(0, graph.size), num_supplier)
    SupplyNodeSet = {}
    for node in supplyNodeList[:-1]:
        SupplyNodeSet[node] = random.randrange(total_supply)
        total_supply -= SupplyNodeSet[node]
    SupplyNodeSet[supplyNodeList[-1]] = total_supply
    #affine_voc.append(VoCAffine(DemandNodeSet,SupplyNodeSet,capacity_edges,A_node_edge_matrix,EdgeSet))
    linear_voc.append(VoCLinear(DemandNodeSet,SupplyNodeSet,capacity_edges,A_node_edge_matrix,EdgeSet))

plt.plot(list_decentralization_degree,affine_voc,label='Affine Supply')
plt.plot(list_decentralization_degree,linear_voc,label='Linear Supply')
plt.xlabel('degree_of_decentralization')
plt.ylabel('VoC')
plt.legend(loc='best')
plt.show() 


# In[ ]:


x = ['a','b']
scale = 3.50
plt.figure(figsize = (15,2))
plt.plot([1,2,3],[1,2,3])
plt.xlabel('Scatter plot \n \n (' + str(x[0]) + ') \n $\gamma$ = %.2f'%scale)
plt.show()


# In[ ]:


#Generate random numbers such that sum is constant
#Suppose distribution is lognormal


# In[ ]:


y = random_fixed_sum(np.log(2),np.log(19),10000)
intended_m = np.log(2)
intended_var = np.log(19)
mean = sum(y)/10000
varrr = (np.var(y))
print(mean,varrr,intended_m,intended_var)


# In[ ]:




