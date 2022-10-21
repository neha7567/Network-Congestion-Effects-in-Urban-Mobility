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
