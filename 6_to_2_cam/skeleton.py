import networkx as nx
'''
Joints
------
0:  BODY_COXA,    :15
1:  COXA_FEMUR,   :16 
2:  FEMUR_TIBIA,  :17
3:  TIBIA_TARSUS, :18
4:  TARSUS_TIP,   :19

5:  BODY_COXA,    :20
6:  COXA_FEMUR,   :21
7:  FEMUR_TIBIA,  :22
8:  TIBIA_TARSUS, :23
9:  TARSUS_TIP,   :24
    
10: BODY_COXA,    :25
11: COXA_FEMUR,   :26
12: FEMUR_TIBIA,  :27
13: TIBIA_TARSUS, :28
14: TARSUS_TIP,   :29
'''

def skeleton():
    edges = [(0,1),(1,2),(2,3),(3,4),
             (5,6),(6,7),(7,8),(8,9),
             (10,11),(11,12),(12,13),(13,14),
             (15,16),(16,17),(17,18),(18,19),
             (20,21),(21,22),(22,23),(23,24),
             (25,26),(26,27),(27,28),(28,29)]

    #0: LF, 1: LM, 2: LH, 3: RF, 4: RM, 5: RH, 
    limb_id = [i for i in range(6) for j in range(5)]
    nodes = [i for i in range(30)]
    
    colors = [[186,30,49], [201,86,79], [213,133,121], #RF, RM, RH
              [15,115, 153], [26,141, 175], [117,190,203] #LF, LM, LH
              ]
    
    edge_colors = [[x / 255.0 for x in colors[i]] for i in limb_id]
    
    #build graph
    G=nx.Graph()
    G.add_edges_from(edges)
    G.add_nodes_from(nodes)
    
    return G, edge_colors