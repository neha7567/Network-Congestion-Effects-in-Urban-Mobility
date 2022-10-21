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
