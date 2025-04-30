from graphviz import Digraph

dot = Digraph(format='png')
dot.attr(rankdir='LR', size='8000,5000')

# Nodes
dot.node('A', 'Airfoil Dataset (.npy)', shape='cylinder')
dot.node('B', 'AirfoilDataset (PyTorch)', shape='box')
dot.node('C', 'DataLoader', shape='box')
dot.node('D', 'Critic Model\n(evaluates real/fake)', shape='box')
dot.node('E', 'Generator Model\n(produces 64x2 points)', shape='box')
dot.node('F', 'Train Loop\n(WGAN-GP)', shape='ellipse')
dot.node('G', 'Generated Airfoils\n(raw 64x2)', shape='box')
dot.node('H', 'Bezier + Normalization', shape='box')
dot.node('I', 'Final Airfoils\n(smooth & scaled)', shape='box')
dot.node('J', 'Save to .npy or\nPlot with Matplotlib', shape='parallelogram')

# Edges
dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'F', label='Batches of real data')
dot.edge('E', 'F', label='Fake samples')
dot.edge('D', 'F', label='Critic scores')
dot.edge('F', 'E', label='Train Generator')
dot.edge('F', 'D', label='Train Critic')
dot.edge('E', 'G', label='Inference')
dot.edge('G', 'H')
dot.edge('H', 'I')
dot.edge('I', 'J')

dot.render('airfoil_gan_flowchart', view=True)
