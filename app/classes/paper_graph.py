import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class PaperGraph(object):
    def __init__(self):
        self.review_preferences_graph = nx.Graph()
        self.total_weights = {}
        self.prob_matrix = []
        self.damping_factor = 0.85
        self.num_iterations = 20

    def add_node(self, paper):
        self.review_preferences_graph.add_node(paper)
        self.total_weights[paper] = 0

    def add_connection(self, paper1, paper2, weight):
        """
        Adds a new connection. If a connection already exists, then the weight is added to the existing connection's
        weight. If the weight is 0, the connection will not be created
        """
        if weight == 0:
            return
        for con in self.review_preferences_graph.edges():
            if (con[0] == paper1 and con[1] == paper2) or (con[0] == paper2 and con[1] == paper1):
                ##con.weight += weight
                self.review_preferences_graph[paper1][paper2]['weight'] += weight
                self.total_weights[paper1] += weight
                self.total_weights[paper2] += weight
                return
        self.review_preferences_graph.add_edge(paper1, paper2, weight=weight)
        self.total_weights[paper1] += weight
        self.total_weights[paper2] += weight

    def connection_weight(self, paper1, paper2):
        """
        returns the weight of the connection, or 0 if the nodes are not connected
        :return: int
        """
        for con in self.review_preferences_graph.edges():
            if (con[0] == paper1 and con[1] == paper2) or (con[0] == paper2 and con[1] == paper1):
                return self.review_preferences_graph[paper1][paper2]['weight']
        return 0

    def create_probability_matrix(self):
        """
        In an unweighted graph, this is simply: M[i][j] = 1/OutboundConnections(j) if j links to i, or 0 otherwise
        In a weighted graph, it is therefore a*(1/OutCon(j)), where a is the weight of the connection from i to j
        Since the weight is the probability, it should be Weightij/Sum(weights of a node)
        :return:
        """
        self.prob_matrix = []
        for nodei in self.review_preferences_graph.nodes():
            probs = []
            for nodej in self.review_preferences_graph.nodes():
                if nodei == nodej:
                    probs.append(0)
                else:
                    if self.total_weights[nodei] == 0:
                        probs.append(0)  # Unconnected nodes would otherwise result in division by 0
                    else:
                        probs.append(self.connection_weight(nodei, nodej) / self.total_weights[nodei])
            self.prob_matrix.append(probs)
        # print("MATRIX: ")
        # print(self.prob_matrix)
        self.prob_matrix = np.array(self.prob_matrix)
        """
        for i in self.nodes:
            print(i, self.total_weights[i])
        for i in self.prob_matrix:
            print(sum(i))
        """

    def get_pr(self, start):
        """
        The pagerank of all papers will be stored in a vector PR[]
            PR[i] = pagerank of i
        Initial PR[i] is defined as 1/N, where N is the number of nodes in a graph (in this case the number of papers)
        The matrix M is the adjencency matrix, and Mij is defined as probability of reaching i from j
        E is a vector of zeroes, except for the value matching the starting node, where it is one
        We can  nx.draw(self.term_graph, node_color='c',edge_color='k', with_labels=True)
        plt.draw()
        plt.show() calculate PR as:
            PR' = (1-d)*E + d*M*PR
        """
        # Probability matrix is stored in self.prob_matrix
        # Isolated nodes (nodes with no connections) need to be treated seperately, since by normal calculation
        #   their pagerank does not sum to 1
        PR = []
        E = []
        row = 0
        for index, node in enumerate(self.review_preferences_graph.nodes()):
            if node == start:
                E.append(1)
                PR.append(1)
                row = index
            else:
                E.append(0)
                PR.append(0)
        if sum(self.prob_matrix.tolist()[row]) < 0.1:
            return np.array(PR)
        # for none in self.nodes:
        #    PR.append(1/len(self.nodes))
        PR = np.array(PR)
        E = np.array(E)
        for iteration in range(1, self.num_iterations):
            PR = E * (1 - self.damping_factor) + ((self.damping_factor) * np.transpose(self.prob_matrix).dot(PR))
        return PR

