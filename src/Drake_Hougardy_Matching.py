#!/usr/bin/env python

import networkx as nx
import random

#Returns maximum maximal weight matching for a given graph
#using greedy approach.
def greedy_maximum_matching(graph):
    matchingGraph = nx.Graph()
    tempGraph=nx.Graph(graph)

    node1=-1
    node2=-1
    maxWeight = 0
    while len(tempGraph.nodes()) > 1:
        maxWeight = float("-inf")
        for (u,v,d) in tempGraph.edges(data=True):
            if d['weight'] > maxWeight:
                maxWeight=d['weight']
                node1=u
                node2=v

        #Add max edge to matching
        matchingGraph.add_nodes_from([node1,node2])
        matchingGraph.add_edge(node1,node2, weight=maxWeight)

        #remove nodes corresponding to matched edge from graph
        tempGraph.remove_nodes_from([node1,node2])

    return matchingGraph;

def graphWeight(graph):
    weight = 0
    for (u,v,d) in graph.edges(data=True):
        weight += d['weight']
    return weight


def edgeIncidentMatching(matching, edges):
    nodes=[]
    for (u,v,d) in edges:
        nodes.extend([u,v])
    return matching.edges(set(nodes), data=False)

def edgeIncidentMatchingWithoutCenter(matching, edges, center):
    (u,v,d) = center
    tempMatching = nx.Graph(matching)
    tempMatching.remove_edge(u,v)
    nodes=[]
    for (u,v,d) in edges:
        nodes.extend([u,v])
    return tempMatching.edges(set(nodes), data=False)

# Returns maximal matching for a given graph. This algorithm takes linear time.
# and does not guarantee to return an approximate or optimal matching of the graph.
def maximal_matching(graph):
    matchingGraph = nx.Graph()
    tempGraph=nx.Graph(graph)

    while len(tempGraph.nodes()) > 1:
        (node1,node2,d) = tempGraph.edges(data=True)[random.randint(0, tempGraph.number_of_edges()-1)];

        #Add edge to matching
        matchingGraph.add_nodes_from([node1,node2])
        matchingGraph.add_edge(node1,node2, weight=d['weight'])

        #remove nodes corresponding to matched edge from graph
        tempGraph.remove_nodes_from([node1, node2])

    return matchingGraph;

# Computes an optimal possible augmentation for the edge node1<-->node2
def getGoodBetaAugmentation(graph, maxNewMatching, center):
    #TODO

    diffEdges = nx.difference(graph, maxNewMatching)
    for (u,v,d) in diffEdges.edges(data=True):
        edgeIncMatching = edgeIncidentMatchingWithoutCenter(maxNewMatching, [(u,v,d)], center);
        winEdge = d['weight'] - graphWeight(edgeIncMatching)



# Computes a better weight matching for a graph given its maximum maximal matching
def improve_matching (graph, maxMatching):
    maxNewMatching = nx.graph(maxMatching)

    # for each edges e belongs to maxMatching, check if there exists a good beta aug in maxNewMatching.
    for (u,v,d) in maxMatching.edges(data=True):
        betaAug = getGoodBetaAugmentation(graph, maxNewMatching, [(u,v,d)]);
        if betaAug != None:
            # Augment maxNewMatching with betaAug
            #TODO
            dummy=None

    return maxNewMatching

if __name__ == "__main__":
    graph = nx.Graph()
    numNodes = 5

    #Add vertices
    for node1 in range(numNodes):
        graph.add_node(node1)

    #Add edges - complete graph
    for node1 in range(numNodes):
        for node2 in range(node1+1, numNodes):
            graph.add_edge(node1, node2, weight=(node1+node2)*2)

    #Display
    print("original graph")
    print graph.edges(data=True)

    t=greedy_maximum_matching(graph)

    print "Maximum Maximal Matching graph"
    print t.edges(data=True)

    t=maximal_matching(graph)

    print "Maximal Matching graph"
    print t.edges(data=True)

    print edgeIncidentMatching(t, [(0,1,{'weight=10'}),(3,1,{'weight=10'})])