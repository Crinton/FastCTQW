import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx

def plot_probability_curves(prob_matrix, times):
    """
    Plot probability distribution over time for each node.
    Args:
        prob_matrix (np.ndarray): N x T matrix of probabilities |\alpha_i(t)|^2
        times (np.ndarray): Array of time points
    """
    n_nodes = prob_matrix.shape[0]
    for i in range(n_nodes):
        plt.plot(times, prob_matrix[i], label=f'Node {i}')
    plt.xlabel('Time (t)')
    plt.ylabel('Probability |\alpha_i(t)|^2')
    plt.title('Probability Evolution Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_state_distribution(prob_array, title="Quantum State Probability Distribution", 
                           xlabel="Node Index", ylabel="Probability |\u03B1_i|^2"):
    """
    Plot the probability distribution of a quantum state across nodes as a bar plot.
    
    Args:
        prob_array (np.ndarray): Array of probabilities |\alpha_i|^2 for each node.
        title (str, optional): Title of the plot. Defaults to "Quantum State Probability Distribution".
        xlabel (str, optional): Label for x-axis. Defaults to "Node Index".
        ylabel (str, optional): Label for y-axis. Defaults to "Probability |\u03B1_i|^2".
    """

    
    plt.figure(figsize=(10, 6))
    sns.barplot(prob_array)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_state_probability(prob_array, title="Quantum State Probability Distribution", 
                          xlabel="Node Index", ylabel="Probability |\u03B1_i|^2"):
    """
    Plot the probability distribution of a quantum state across nodes as a curve.
    
    Args:
        prob_array (np.ndarray): Array of probabilities |\alpha_i|^2 for each node.
        title (str, optional): Title of the plot. Defaults to "Quantum State Probability Distribution".
        xlabel (str, optional): Label for x-axis. Defaults to "Node Index".
        ylabel (str, optional): Label for y-axis. Defaults to "Probability |\u03B1_i|^2".
    """
    n_nodes = len(prob_array)
    indices = np.arange(n_nodes)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=indices, y=prob_array, color='blue', linewidth=2, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_graph_probability(adj_matrix, probs, title=''):
    """
    Plot graph with node sizes proportional to probabilities.
    Args:
        adj_matrix (np.ndarray): N x N adjacency matrix
        probs (np.ndarray): Probability vector of length N
        title (str): Plot title
    """
    G = nx.from_numpy_array(adj_matrix)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_size=probs * 1000, node_color=probs, cmap='Blues', with_labels=True)
    plt.title(title)
    plt.show()


# # Demo
# n_nodes = 4
# times = np.linspace(0, 2, 20)
# prob_matrix = np.abs(np.random.rand(n_nodes, len(times))) ** 2  # Placeholder data
# plot_probability_curves(prob_matrix, times)

# n_nodes = 4
# adj_matrix = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]])
# probs = np.abs(np.random.rand(n_nodes)) ** 2  # Placeholder data
# plot_graph_probability(adj_matrix, probs, 'Graph Probability at t=0')