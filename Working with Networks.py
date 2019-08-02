import matplotlib.pyplot as plt
import networkx as nx

# Draw the graph to screen
nx.draw(T_sub)
plt.show()
###Basic Querying
noi = [n for n, d in T.nodes(data=True) if d['occupation'] == 'scientist']
eoi = [(u, v) for u, v, d in T.edges(data=True) if d['date'] < date(2010, 1, 1)]

T.edges[1, 10]['weight'] = 2
for u, v, d in T.edges(data=True):
    if 293 in [u, v]:
        # Set the weight to 1.1
        T.edges[u, v]['weight'] = 1.1
   
###Self Loops     
def find_selfloop_nodes(G):
    """
    Finds all nodes that have self-loops in the graph G.
    """
    nodes_in_selfloops = []
    for u, v in G.edges():
        if u==v:
            nodes_in_selfloops.append(u)

    return nodes_in_selfloops
assert T.number_of_selfloops() == len(find_selfloop_nodes(T))

###Visuzalization
#With Matrix
import nxviz as nv
m = nv.MatrixPlot(T)
m.draw()
plt.show()
A = nx.to_numpy_matrix(T)
T_conv = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
for n, d in T_conv.nodes(data=True):
    assert 'category' not in d.keys()
    
#Circos Plot
import matplotlib.pyplot as plt
from nxviz import CircosPlot
c = CircosPlot(T)
c.draw()
plt.show()

#Arcos Plot
import matplotlib.pyplot as plt
from nxviz import ArcPlot
a = ArcPlot(T)
a.draw()
plt.show()
a2 = ArcPlot(T, node_order='category', node_color='category')
a2.draw()
plt.show()

###Centrality and other important nodes
def nodes_with_m_nbrs(G, m):
    """
    Returns all nodes in graph G that have m neighbors.
    """
    nodes = set()
    for n in G.nodes():
        if len(list(G.neighbors(n))) == m:
            nodes.add(n)
    return nodes

six_nbrs = nodes_with_m_nbrs(T, 6)
print(six_nbrs)
#other degree
degrees = [len(list(T.neighbors(n))) for n in T.nodes()]

#Degree centrality
import matplotlib.pyplot as plt
deg_cent = nx.degree_centrality(T)
plt.figure()
plt.hist(list(deg_cent.values()))
plt.show()
plt.figure()
plt.hist(degrees)
plt.show()
plt.figure()
plt.scatter(degrees, list(deg_cent.values()))
plt.show()

####Paths in Graph
#BFS
def path_exists(G, node1, node2):
    """
    This function checks whether a path exists between two nodes (node1, node2) in graph G.
    """
    visited_nodes = set()
    queue = [node1]

    for node in queue:
        neighbors = G.neighbors(node)
        if node2 in neighbors:
            print('Path exists between nodes {0} and {1}'.format(node1, node2))
            return True
            break

        else:
            visited_nodes.add(node)
            queue.extend([n for n in neighbors if n not in visited_nodes])
        if node == queue[-1]:
            print('Path does not exist between nodes {0} and {1}'.format(node1, node2))
            return False

#Betweeness Centrality Shortest Path through the node/All shortest paths
            # Compute the betweenness centrality of T: bet_cen
bet_cen = nx.betweenness_centrality(T)
deg_cen = nx.degree_centrality(T)
plt.scatter(list(bet_cen.values()), list(deg_cen.values()))
plt.show()

###Showcase with twitter
# Define find_node_with_highest_bet_cent()
def find_node_with_highest_bet_cent(G):
    bet_cent = nx.betweenness_centrality(G)
    max_bc = max(list(bet_cent.values()))

    nodes = set()
    for k, v in bet_cent.items():
        if v == max_bc:
            nodes.add(k)

    return nodes
top_bc = find_node_with_highest_bet_cent(T)
for node in top_bc:
    assert nx.betweenness_centrality(T)[node] == max(nx.betweenness_centrality(T).values())
    
###Cliques and finding triangle connections
from itertools import combinations
def is_in_triangle(G, n):
    """
    Checks whether a node `n` in graph `G` is in a triangle relationship or not.
    Returns a boolean.
    """
    in_triangle = False
    for n1, n2 in combinations(G.neighbors(n), 2):
        if G.has_edge(n1, n2):
            in_triangle = True
            break
    return in_triangle

def nodes_in_triangle(G, n):
    """
    Returns the nodes in a graph `G` that are involved in a triangle relationship with the node `n`.
    """
    triangle_nodes = set([n])
    for n1, n2 in combinations(G.neighbors(n), 2):
        if G.has_edge(n1, n2):
            triangle_nodes.add(n1)
            triangle_nodes.add(n2)
    return triangle_nodes
assert len(nodes_in_triangle(T, 1)) == 35

from itertools import combinations

# Define node_in_open_triangle()
def node_in_open_triangle(G, n):
    """
    Checks whether pairs of neighbors of node `n` in graph `G` are in an 'open triangle' relationship with node `n`.
    """
    in_open_triangle = False

    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):

        # Check if n1 and n2 do NOT have an edge between them
        if not G.has_edge(n1, n2):

            in_open_triangle = True

            break

    return in_open_triangle
#Find open triangles
num_open_triangles = 0
for n in T.nodes():
    if node_in_open_triangle(T, n):
        num_open_triangles += 1

print(num_open_triangles)

###Finding Maximum cluques
def maximal_cliques(G, size):
    """
    Finds all maximal cliques in graph `G` that are of size `size`.
    """
    mcs = []
    for clique in nx.find_cliques(G):
        if len(clique) == size:
            mcs.append(clique)
    return mcs
assert len(maximal_cliques(T, 3)) == 33

###Subgraphs
nodes_of_interest = [29, 38, 42]  # provided.

# Define get_nodes_and_nbrs()
def get_nodes_and_nbrs(G, nodes_of_interest):
    """
    Returns a subgraph of the graph `G` with only the `nodes_of_interest` and their neighbors.
    """
    nodes_to_draw = []
    for n in nodes_of_interest:
        nodes_to_draw.append(n)
        for nbr in G.neighbors(n):
            nodes_to_draw.append(nbr)

    return G.subgraph(nodes_to_draw)
T_draw = get_nodes_and_nbrs(T, nodes_of_interest)
nx.draw(T_draw)
plt.show()

# Extract the nodes of interest: nodes
nodes = [n for n, d in T.nodes(data=True) if d['occupation'] == 'celebrity']
nodeset = set(nodes)
for n in nodes:

    nbrs = T.neighbors(n)
    nodeset = nodeset.union(nbrs)
T_sub = T.subgraph(nodeset)

# Draw T_sub to the screen
nx.draw(T_sub)
plt.show()


###CASE STUDY
import matplotlib.pyplot as plt
import networkx as nx
plt.hist(list(nx.degree_centrality(G).values()))
plt.show()

plt.hist(list(nx.betweenness_centrality(G).values()))
plt.show()

#Visualize
from nxviz import MatrixPlot
import matplotlib.pyplot as plt
largest_ccs = sorted(nx.connected_component_subgraphs(G), key=lambda x: len(x))[-1]

h = MatrixPlot(graph=largest_ccs, node_grouping='grouping')
h.draw()
plt.show()


from nxviz.plots import ArcPlot
for n, d in G.nodes(data=True):
    G.node[n]['degree'] = nx.degree(G, n)

a = ArcPlot(graph=G, node_order='degree')
a.draw()
plt.show()

# Import necessary modules
from nxviz import CircosPlot
for n, d in G.nodes(data=True):
    G.node[n]['degree'] = nx.degree(G, n)

c = CircosPlot(G,node_grouping = 'grouping',node_color = 'grouping', node_order='degree')
c.draw()
plt.show()


###Cliques
import networkx as nx
from nxviz import CircosPlot
import matplotlib.pyplot as plt
largest_clique = sorted(nx.find_cliques(G), key=lambda x:len(x))[-1]
G_lc = G.subgraph(largest_clique)
c = CircosPlot(G_lc)

c.draw()
plt.show()

###Find Important Users
deg_cent = nx.degree_centrality(G)
max_dc = max(deg_cent.values())
prolific_collaborators = [n for n, dc in deg_cent.items() if dc == max_dc]
print(prolific_collaborators)

###
from nxviz import ArcPlot
import matplotlib.pyplot as plt
largest_max_clique = set(sorted(nx.find_cliques(G), key=lambda x: len(x))[-1])
G_lmc = G.subgraph(largest_max_clique).copy()

# Go out 1 degree of separation
for node in list(G_lmc.nodes()):
    G_lmc.add_nodes_from(G.neighbors(node))
    G_lmc.add_edges_from(zip([node]*len(list(G.neighbors(node))), G.neighbors(node)))

for n in G_lmc.nodes():
    G_lmc.node[n]['degree centrality'] = nx.degree_centrality(G_lmc)[n]

a = ArcPlot(G_lmc, node_order='degree centrality')
a.draw()
plt.show()

###Recomend for non connected
# Import necessary modules
from itertools import combinations
from collections import defaultdict

# Initialize the defaultdict: recommended
recommended = defaultdict(int)

# Iterate over all the nodes in G
for n, d in G.nodes(data=True):

    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):

        # Check whether n1 and n2 do not have an edge
        if not G.has_edge(n1, n2):

            # Increment recommended
            recommended[(n1, n2)] += 1

# Identify the top 10 pairs of users
all_counts = sorted(recommended.values())
top10_pairs = [pair for pair, count in recommended.items() if count > all_counts[-10]]
print(top10_pairs)
