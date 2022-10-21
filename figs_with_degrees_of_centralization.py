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
