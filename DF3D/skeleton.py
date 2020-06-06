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
0:  BODY_COXA,    :19 
1:  COXA_FEMUR,   :20 
2:  FEMUR_TIBIA,  :21
3:  TIBIA_TARSUS, :22
4:  TARSUS_TIP,   :23

5:  BODY_COXA,    :24
6:  COXA_FEMUR,   :25
7:  FEMUR_TIBIA,  :26
8:  TIBIA_TARSUS, :27
9:  TARSUS_TIP,   :28
    
10: BODY_COXA,    :29
11: COXA_FEMUR,   :30
12: FEMUR_TIBIA,  :31
13: TIBIA_TARSUS, :32
14: TARSUS_TIP,   :33

15: ANTENNA,      :34
16: STRIPE,       :35
17: STRIPE,       :36
18: STRIPE,       :37
'''

def skeleton():
    edges = [(0,1),(1,2),(2,3),(3,4),
             (5,6),(6,7),(7,8),(8,9),
             (10,11),(11,12),(12,13),(13,14),
             (19,20),(20,21),(21,22),(22,23),
             (24,25),(25,26),(26,27),(27,28),
             (29,30),(30,31),(31,32),(32,33)]

    #0: LF, 1: LM, 2: LH, 3: RF, 4: RM, 5: RH, 
    limb_id = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
    
    nodes = [0, 1, 2, 3, 4,
             5, 6, 7, 8, 9,
             10, 11, 12, 13, 14, 
             19, 20, 21, 22, 23, 
             24, 25, 26, 27, 28, 
             29, 30, 31, 32, 33]
    
    colors = [[186,30,49], [201,86,79], [213,133,121], #RF, RM, RH
              [15,115, 153], [26,141, 175], [117,190,203] #LF, LM, LH
              ]
    
    edge_colors = [[x / 255.0 for x in colors[i]]  for i in limb_id]
    
    #build graph
    G=nx.Graph()
    G.add_edges_from(edges)
    G.add_nodes_from(nodes)
    
    return G, edge_colors