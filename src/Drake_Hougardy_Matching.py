#!/usr/bin/env python

import networkx as nx
import random
from datetime import datetime
import sys, traceback

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

# Returns maximal matching for a given graph. This algorithm takes linear time.
# and does not guarantee to return an approximate or optimal matching of the graph.
def maximal_matching(graph):
    matchingGraph = nx.Graph()
    tempGraph=nx.Graph(graph)
    edges=tempGraph.edges(data=True)
    numEdges=len(edges)
    while numEdges > 0:
        randomNum=random.randint(0, numEdges-1)
        (node1,node2,d) = edges[randomNum];

        #Add edge to matching
        matchingGraph.add_nodes_from([node1,node2])
        matchingGraph.add_edge(node1,node2, weight=d['weight'])

        #remove nodes corresponding to matched edge from graph
        tempGraph.remove_nodes_from([node1, node2])
        edges=tempGraph.edges(data=True)
        numEdges=len(edges)

    return matchingGraph;

#Augments new edges to the matching and removes old conflicting edges in matching.
def augmentMatching(graph, matching, newEdges):
    for (u,v,d) in newEdges:
        adjNodes=[u,v]
        if matching.has_node(u):
            adjNodes.append(matching.neighbors(u)[0])

        if matching.has_node(v):
            adjNodes.append(matching.neighbors(v)[0])

        matching.remove_nodes_from(set(adjNodes))
        matching.add_edge(u,v,graph.get_edge_data(u,v))

    return matching

#Returns weight of the graph
def graphWeight(graph):
    return edgeListWeight(graph.edges(data=True))

#Returns the sum of all weights of the provided edges.
def edgeListWeight(ebunch):
    weight = 0
    for (u,v,d) in ebunch:
        weight += d['weight']
    return weight

#Returns all the edges adjacent to 'edges' in matching. If a 'center' edge is given,
# the method by default includes the weight of the 'center' edge if it is in matching
# else does not include it. This feature can be changed by modifying 'includeCentre' argument.
def edgeIncidentMatching(matching, edges, center=None, includeCentre=True):
    edgeList=[]
    computedEdges={}
    weight=0
    for (u,v,d) in edges:
        for node in [u,v]:
            if matching.has_node(node) :
                for k, kd in matching[node].iteritems():
                    key=str(sorted([k,node]))
                    if(not computedEdges.has_key(key)):
                        computedEdges[key]=1
                        edgeList.append((node, k, kd))
                        weight += kd['weight']
    if not includeCentre:
        (x,y,dxy) = center
        if matching.has_edge(x,y):
            weight=weight-dxy['weight']
    return (edgeList, weight)

#Returns all the edges adjacent to 'edges' in matching. Includes the weight of the
# 'center' edge if it is in matching else it is not included.
def edgeIncidentMatchingWithCenter(matching, edges, center):
    return edgeIncidentMatching(matching,edges,center,True)

# Returns top two edges with maximum surplus values
# edges have 'surplus' attributes as well. Please refer maxAllowable method
def getMaxTwoSurplusEdges(edges):
    edge1=None
    edge2=None

    max1=0
    max2=0
    for (u,v,d) in edges:
        surplus= d['surplus']

        if surplus > max1:
            max2=max1
            max1=surplus

            edge2=edge1
            edge1=(u,v,d)
        elif surplus > max2:
            max2=surplus
            edge2=(u,v,d)

    return (edge1, edge2)

#Returns the edgelist contributing maximum surplus with 'edge' and the edge in 'edgeList'
# for the 'center' edge, if any.
def getMaxNonAdjacentSurplusEdges(matching, edgeList, edge, center , betaValue):
    maxWinWeight=0;
    z=0
    (x,y,dxy)=center
    if(matching.has_edge(x,y)):
        z=dxy['weight']

    (m,n,dmn)=edge
    edge1diff = betaValue * z - (dmn['surplus'])

    surplusEdgeList1=[]

    for (u,v,d) in edgeList:
        #Check for non-adjacency with edge
        if not (u in [m,n] or v in [m,n]):
            if(d['surplus'] >= edge1diff) :
                tempList=[(u,v,d), (m,n,dmn)]
                (edgeIncMatchingWC, edgeIncWeight) = edgeIncidentMatchingWithCenter(matching, tempList, center);
                winEdge1 =  dmn['weight'] + d['weight']-edgeIncWeight
                if(winEdge1 > maxWinWeight):
                    surplusEdgeList1=tempList
                    maxWinWeight=winEdge1

    return (surplusEdgeList1, maxWinWeight)

#Returns the two edges adjacent to 'center' edge that contributes maximum positive surplus, if any.
#The returned edges{a,b} does not form cycle with {a,b,center} U M(a) U M(b)
def maxAllowable(matching, edges, center, betaValue):
    (x,y,dxy)=center
    #Edge with maximum surplus
    maxEdges=None
    maxWinValue=0

    #All edges adjacent to node 'x'
    xEdgeList=[]
    #All edges adjacent to node 'y'
    yEdgeList=[]

    for (u,v,d) in edges:
        #Calculate surplus
        surplus= d['surplus']

        #TODO Check if we need to ensure this edge is not center
        if u==x or v==x :
            xEdgeList.append((u,v,d))
        elif u==y or v==y :
            yEdgeList.append((u,v,d))
        else :
            #TODO throw exception
            dummy=None

    #Find top two edges with maximum surplus
    (xEdge1, xEdge2) = getMaxTwoSurplusEdges(xEdgeList)
    (yEdge1, yEdge2) = getMaxTwoSurplusEdges(yEdgeList)

    for edge in (xEdge1, xEdge2):
        if edge!=None:
            xAugList, xWinEdgeValue = getMaxNonAdjacentSurplusEdges(matching, yEdgeList, edge, center, betaValue)
            if xWinEdgeValue>maxWinValue :
                maxWinValue = xWinEdgeValue;
                maxEdges=xAugList

    for edge in (yEdge1, yEdge2):
        if edge!=None:
            yAugList, yWinEdgeValue = getMaxNonAdjacentSurplusEdges(matching, xEdgeList, edge, center, betaValue)
            if yWinEdgeValue>maxWinValue :
                maxWinValue = yWinEdgeValue;
                maxEdges=yAugList

    return (maxEdges, maxWinValue)

def getNextBetaValue(currentWeight):
    newBetaWeight= (4.0+9.0*currentWeight*(4.0+currentWeight))/48.0
    newBetaValue= 4.0/(2.0 + 3.0* newBetaWeight)

    return (newBetaWeight, newBetaValue)

# Computes an optimal possible augmentation for the edge node1<-->node2
# Here we consider all edges a belongs to E\ M that is adjacent with center., the win
def getGoodBetaAugmentation(graph, maxNewMatching, center , betaValue):
    (node1, node2, dxy) = center
    centerWeightInMatching = (0,dxy['weight'])[maxNewMatching.has_edge(node1,node2)]

    localMax= 0
    maxBetaAugmentation=None
    maxBetaAugmentationWeight=localMax

    #Best augmentation with atmost one edge
    aug1=[]
    aug1WinValue=localMax;

    #Best augmentation with a cycle
    aug2=[];
    aug2WinValue=localMax

    #Adding win, surplus and incweight attributes to all edges in E-M
    winWeightedEdges=[]

    # Find Aug Set with atmost one edge
    for (u,v,d) in graph.edges([node1, node2], data=True):
        # Skip if the edges belongs to matching
        if(not maxNewMatching.has_edge(u,v)):
            (edgeIncMatchingWC, edgeIncWeight) = edgeIncidentMatchingWithCenter(maxNewMatching, [(u,v,d)], center);
            tempWinEdge = d['weight'] - edgeIncWeight

            #New dictionary to hold extra attributes
            du=dict(d)
            du['win'] = tempWinEdge
            du['surplus'] = du['weight'] - betaValue*(edgeIncWeight-centerWeightInMatching)
            winWeightedEdges.append((u,v,du))

            #Check if the edge has maximum win
            if(tempWinEdge>aug1WinValue) :
                aug1=[(u,v,d)]
                aug1WinValue=tempWinEdge

    if(aug1WinValue > maxBetaAugmentationWeight) :
        maxBetaAugmentation = aug1
        maxBetaAugmentationWeight=aug1WinValue

    #Find Matching with 2 edges and forming cycle
    # Case-1: When M(a) contains an edge adjacent to center.
    if (maxNewMatching.has_node(node1)and maxNewMatching.has_node(node2) and not maxNewMatching.has_edge(node1,node2)) :
        centerMatchEdge1 = maxNewMatching[node1]
        centerMatchEdge2 = maxNewMatching[node2]
        u=None

        key1 = centerMatchEdge1.keys()[0]
        key2 = centerMatchEdge2.keys()[0]
        if(graph.has_edge(node2, key1)) :
            d = centerMatchEdge1[key1]
            u = node1;
            aug2.append((node2, key1, d))

        elif(graph.has_edge(node1, key2)) :
            d = centerMatchEdge2[key2]
            u = node2;
            aug2.append((node1, key2, d))

        if(bool(aug2)):
            (p,q,dpq) = aug2[0]
            aWeight=dpq['weight'];
            bTemp=None
            for v, d in graph[u].iteritems() :
                if v != p and v!= q:
                    (edgeIncMatchingWC, edgeIncWeight) = edgeIncidentMatchingWithCenter(maxNewMatching, [aug2[0], (u,v,d)], center);
                    tempWinEdge = d['weight'] +aWeight - edgeIncWeight
                    if tempWinEdge>aug2WinValue :
                        bTemp = (u,v,d)
                        aug2WinValue=tempWinEdge

            if bool(bTemp):
                aug2.append(bTemp);

    # Case-2:  M(a) intersection M(b) contains an edge not incident with an end vertex of center
    aPossibleEdges = graph[node1]
    for v, d in aPossibleEdges.iteritems():
        if( v != node2 and not maxNewMatching.has_edge(node1,v) ): #Exclude the center edge and the new edge 'a', Read paper Lemma-2
            if maxNewMatching.has_node(v) :
                bNode=maxNewMatching.neighbors(v)[0]
                if bNode!=node2 and graph.has_edge(node2, bNode):
                    #Check Beta Augmentation
                    bNodeData=graph.get_edge_data(node2, bNode)
                    (edgeIncMatchingWC, edgeIncWeight) = edgeIncidentMatchingWithCenter(maxNewMatching, [(node1, v, d), (node2, bNode, bNodeData)], center)
                    tempWinEdge = d['weight'] + bNodeData['weight'] - edgeIncWeight
                    if(tempWinEdge>aug2WinValue) :
                        aug2=[(node1, v, d), (node2, bNode, bNodeData)]
                        aug2WinValue=tempWinEdge

    if(aug2WinValue > maxBetaAugmentationWeight) :
        maxBetaAugmentation = aug2
        maxBetaAugmentationWeight = aug2WinValue

    # Find augmentation of two edges a, b such that {a, b, e} U M(a) does not contain a cycle.
    # Get edges with win(u,v) > 1/2* w(u,v)
    tempWinEdges=[]
    for(u,v,d) in winWeightedEdges:
        if(d['win']>= 0.5*dxy['weight']) :
            tempWinEdges.append((u,v,d))
    augTemp, augTempWinValue = maxAllowable(maxNewMatching, tempWinEdges, center, betaValue)

    if augTempWinValue > maxBetaAugmentationWeight :
        maxBetaAugmentationWeight = augTempWinValue
        maxBetaAugmentation=augTemp

    augTemp, augTempWinValue = maxAllowable(maxNewMatching, winWeightedEdges, center, betaValue)
    if augTempWinValue > maxBetaAugmentationWeight :
        maxBetaAugmentationWeight = augTempWinValue
        maxBetaAugmentation=augTemp

    return maxBetaAugmentation, maxBetaAugmentationWeight

# Computes a better weight matching for a graph given its maximum maximal matching
def improve_matching (graph, maxMatching, betaValue):

    #Make maxMatching Maximal
    matchingNodesSet = set(maxMatching.nodes())
    graphNodesSet = set(graph.nodes())
    differenceNodes = graphNodesSet - matchingNodesSet
    if(differenceNodes > 1) :
        tempMatching = maximal_matching(graph.subgraph(differenceNodes))
        maxMatching.add_edges_from(tempMatching.edges(data=True))

    # Copy matching to new Matching M'
    maxNewMatching = nx.Graph(maxMatching)

    # for each edges e belongs to maxMatching, check if there exists a good beta aug in maxNewMatching.
    for (u,v,d) in maxMatching.edges(data=True):
        betaAug, betaAugWeight = getGoodBetaAugmentation(graph, maxNewMatching, (u,v,d), betaValue);

        if betaAug != None :
            #print "Beta Aug Selected: "+str(betaAug)
            #print "Beta Aug Weight: "+str(betaAugWeight)
            #print "Weight of Edges: "+str(edgeIncidentMatching(maxNewMatching, betaAug)[1])
            
            # Augment maxNewMatching with betaAug
            augmentMatching(graph, maxNewMatching, betaAug)

    return maxNewMatching

def weight_of_matching(graph, matching):
    weight = 0
    for x, y in matching.items():
        weight += graph.edge[x][y]["weight"]
    return weight / 2

#TODO - Test this module
def multi_improve_matching(graph, exactMaxMatchingWeight=None):
    matching = maximal_matching(graph);
    currentBetaWeight = 0.5
    iteration = 0

    #Get BetaValue
    newBetaWeight, betaValue=getNextBetaValue(currentBetaWeight)
    currMatchingWeight = graphWeight(matching)

    while currMatchingWeight/exactMaxMatchingWeight < 0.67:
        iteration += 1
        print "Befre Improve Matching weight: " + str(currMatchingWeight)
        matching = improve_matching(graph, matching, betaValue)
        currMatchingWeight = graphWeight(matching)

        print "After Improve Matching weight: " + str(currMatchingWeight)
        print "iteration: "+ str(iteration)
        print "Ratio: "+ str(currMatchingWeight/exactMaxMatchingWeight)

        currentBetaWeight = newBetaWeight
        newBetaWeight, betaValue=getNextBetaValue(currentBetaWeight)

    return matching

if __name__ == "__main__":
        graph = nx.Graph()
        numNodes = 100
        betaValue=1.25

        #Add vertices
        for node1 in range(numNodes):
            graph.add_node(node1)

        #Add edges - complete graph
        for node1 in range(numNodes):
            for node2 in range(node1+1, numNodes):
                graph.add_edge(node1, node2, weight=random.uniform(5,40))
                #graph.add_weighted_edges_from([(0,1,13.0),(1,2,7),(2,0,15), (3,0,12), (3,4,20), (4,0,21) ])

        #Display
        print("original graph")
        #print graph.edges(data=True)

        #t=maximal_matching(graph);
        #print "maximal matching weight" + str(graphWeight(t))
        #print t.edges(data=True)

        exact_matching = nx.max_weight_matching(graph)
        exactMaxMatchingWeight = weight_of_matching(graph, exact_matching)
        print "Exact Matching weight" + str(exactMaxMatchingWeight)

        #b=improve_matching(graph, t, betaValue)
        startTime = datetime.now()
        b=multi_improve_matching(graph, exactMaxMatchingWeight)
        elapsedTime = datetime.now() - startTime
        improvedWeight = graphWeight(b)
        print "Improved weight" + str(improvedWeight)
        print "Improved Ratio  : " + str(improvedWeight/exactMaxMatchingWeight)
        print "Time taken(sec) : " + str(elapsedTime.seconds + elapsedTime.microseconds/1E6)
        #print b.edges(data=True)


        startTime = datetime.now()
        t=greedy_maximum_matching(graph)
        elapsedTime = datetime.now() - startTime
        greedyWeight = graphWeight(t);
        print "\nGreedy weight  : " + str(greedyWeight)
        print "Greedy Ratio  : " + str(greedyWeight/exactMaxMatchingWeight)
        print "Time taken(sec) : " + str(elapsedTime.seconds + elapsedTime.microseconds/1E6)
        #print "Maximum Maximal Matching graph"
        #print t.edges(data=True)
