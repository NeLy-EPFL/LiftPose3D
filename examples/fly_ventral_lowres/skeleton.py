#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:08:38 2020

@author: adamgosztolai
"""

import networkx as nx
'''
Joints
------
0:  COXA_FEMUR,   :12
1:  FEMUR_TIBIA,  :13
2:  TIBIA_TARSUS, :14
3:  TARSUS_TIP,   :15

4:  COXA_FEMUR,   :16
5:  FEMUR_TIBIA,  :17
6:  TIBIA_TARSUS, :18
7:  TARSUS_TIP,   :19
    
8:  COXA_FEMUR,   :20
9:  FEMUR_TIBIA,  :21
10: TIBIA_TARSUS, :22
11: TARSUS_TIP,   :24
'''

def skeleton():
    edges = [(0,1),(1,2),(2,3),
             (4,5),(5,6),(6,7),
             (8,9),(9,10),(10,11),
             (12,13),(13,14),(14,15),
             (16,17),(17,18),(18,19),
             (20,21),(21,22),(22,23)]

    #0: LF, 1: LM, 2: LH, 3: RF, 4: RM, 5: RH, 
    limb_id = [i for i in range(6) for j in range(4)]
    nodes = [i for i in range(24)]
    
    colors = [[15,115,153], [26,141,175], [117,190,203], #LF, LM, LH
              [186,30,49], [201,86,79], [213,133,121] #RF, RM, RH
              ] 
    
    edge_colors = [[x / 255.0 for x in colors[i]] for i in limb_id]
    
    #build graph
    G=nx.Graph()
    G.add_edges_from(edges)
    G.add_nodes_from(nodes)
    
    return G, edge_colors