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





