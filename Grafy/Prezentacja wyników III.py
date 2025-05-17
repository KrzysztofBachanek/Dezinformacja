import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import imageio.v2 as imageio
import os
import matplotlib.ticker as ticker

# PARAMETRY SYMULACJI
NUM_AGENTS = 500 # całkowita liczba osób 

NUM_INFLUENCERS = 5 # liczba influencerów
NUM_BOTS = 50 # liczba botów
NUM_SCEPTICS = 250 # liczba sceptyków

INFLUENCER_DEGREE = 100 # liczba połączeń dla influencerów
BOT_DEGREE = 10 # liczba połączeń dla botów

NUM_INITIAL_INFECTED = 3 # początkowa liczba zainfekowanych
SIMULATION_STEPS = 20 # liczba kroków czasowych
GIF_FRAME_DURATION = 2.0  # sekundy na klatkę

class Agent:
    def __init__(self, id, agent_type='normal'):
        self.id = id
        self.agent_type = agent_type
        self.infected = False
        self.set_behavior()

    def set_behavior(self):
        
        if self.agent_type == 'normal':
            self.spreading_prob = 0.5
            self.resistance = 0.5
        
        elif self.agent_type == 'influencer':
            self.spreading_prob = 0.7
            self.resistance = 0.3
        
        elif self.agent_type == 'bot':
            self.spreading_prob = 1.0
            self.resistance = 0.0
        
        elif self.agent_type == 'skeptic':
            self.spreading_prob = 0.1
            self.resistance = 0.9

    def try_infect(self, neighbors):
        if np.random.rand() < self.spreading_prob:
            for neighbor in neighbors:
                if not neighbor.infected and np.random.rand() > neighbor.resistance:
                    neighbor.infected = True

# TWORZENIE SIECI Z KONTROLĄ TYPOW I POŁĄCZEŃ
def create_custom_network(n, num_influencers, influencer_degree, num_bots, bot_degree, num_sceptics):
    G = nx.Graph()
    agents = {}

    all_ids = list(range(n))
    random.shuffle(all_ids)

    influencer_ids = all_ids[:num_influencers]
    bot_ids = all_ids[num_influencers:num_influencers + num_bots]
    skeptic_ids = all_ids[num_influencers + num_bots:num_influencers + num_bots + num_sceptics]
    normal_ids = all_ids[num_influencers + num_bots + num_sceptics:]

    for i in influencer_ids:
        agents[i] = Agent(i, agent_type='influencer')
        G.add_node(i)

    for i in bot_ids:
        agents[i] = Agent(i, agent_type='bot')
        G.add_node(i)

    for i in skeptic_ids:
        agents[i] = Agent(i, agent_type='skeptic')
        G.add_node(i)

    for i in normal_ids:
        agents[i] = Agent(i, agent_type='normal')
        G.add_node(i)
    
    # Influencer connections
    for i in influencer_ids:
        possible_targets = [nid for nid in G.nodes() if nid != i and not G.has_edge(i, nid)]
        targets = random.sample(possible_targets, min(influencer_degree, len(possible_targets)))
        for t in targets:
            G.add_edge(i, t)
    
    # Bot connections
    for i in bot_ids:
        possible_targets = [nid for nid in G.nodes() if nid != i and not G.has_edge(i, nid)]
        targets = random.sample(possible_targets, min(bot_degree, len(possible_targets)))
        for t in targets:
            G.add_edge(i, t)
    
    # Connect others with degree 3
    for i in skeptic_ids + normal_ids:
        while G.degree[i] < 3:
            t = random.choice(list(G.nodes()))
            if t != i and not G.has_edge(i, t):
                G.add_edge(i, t)
    return G, agents

# SYMULACJA ROZPRZESTRZENIANIA DEZINFORMACJI
def simulate_spread(G, agents, steps, initial_infected_count):
    
    initial_infected = random.sample(list(agents.keys()), initial_infected_count)
    for i in initial_infected:
        agents[i].infected = True

    typewise_over_time = []
    frames = []
    frames.append(draw_graph(G, agents, 0))
    
    for step in range(1, steps+1):
        counts = {'normal': 0, 'influencer': 0, 'bot': 0, 'skeptic': 0}
        for a in agents.values():
            if a.infected:
                counts[a.agent_type] += 1
        typewise_over_time.append(counts)
        
        for node in G.nodes():
            if agents[node].infected:
                neighbors = [agents[n] for n in G.neighbors(node)]
                agents[node].try_infect(neighbors)

        frames.append(draw_graph(G, agents, step))
    counts = {'normal': 0, 'influencer': 0, 'bot': 0, 'skeptic': 0}
    for a in agents.values():
        if a.infected:
            counts[a.agent_type] += 1
    typewise_over_time.append(counts)
    return typewise_over_time, frames

# RYSOWANIE GRAFÓW
def draw_graph(G, agents, step):
    plt.figure(figsize=(16, 10))
    pos = nx.spring_layout(G, seed=42, k=0.6)
    colors = []
    for node in G.nodes():
        agent = agents[node]
        if agent.infected:
            colors.append('red')
        elif agent.agent_type == 'bot':
            colors.append('blue')
        elif agent.agent_type == 'influencer':
            colors.append('yellow')
        elif agent.agent_type == 'skeptic':
            colors.append('gray')
        else:
            colors.append('green')
    
    nx.draw(G, pos, node_color=colors, edgecolors='black', with_labels=False, node_size=150)
    
    plt.text(0.95, 0.95, f"Step {step}", fontsize=14, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.3'))

    plt.axis('off')
    filename = f"frames/frame_{step:03}.png"
    os.makedirs("frames", exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    return filename
# GENEROWANIE GIF-a 
def create_gif(frames, output_file="Dezinformacja1.gif"):
    images = [imageio.imread(frame) for frame in frames]
    imageio.mimsave(output_file, images, duration=GIF_FRAME_DURATION)
    for frame in frames:
        os.remove(frame)
    os.rmdir("frames")

# URUCHOMIENIE SYMULACJI
G, agents = create_custom_network(
    n = NUM_AGENTS, 
    num_influencers = NUM_INFLUENCERS, 
    influencer_degree = INFLUENCER_DEGREE, 
    num_bots = NUM_BOTS, 
    bot_degree = BOT_DEGREE,
    num_sceptics = NUM_SCEPTICS)

typewise_over_time, frames = simulate_spread(
    G, agents,
    steps = SIMULATION_STEPS,
    initial_infected_count = NUM_INITIAL_INFECTED
)
create_gif(frames)

# WYKRES: ZMIANA LICZBY ZAKAŻONYCH WG TYPU
steps = list(range(SIMULATION_STEPS+1))
normal = [entry['normal'] for entry in typewise_over_time]
influencer = [entry['influencer'] for entry in typewise_over_time]
bot = [entry['bot'] for entry in typewise_over_time]
skeptic = [entry['skeptic'] for entry in typewise_over_time]

plt.figure(figsize=(10, 6))
plt.plot(steps, normal, marker = 'o', label='Normal', color='green')
plt.plot(steps, influencer, marker = 'o', label='Influencer', color='orange')
plt.plot(steps, bot, marker = 'o', label='Bot', color='blue')
plt.plot(steps, skeptic, marker = 'o', label='Skeptic', color='gray')
plt.xlabel("Krok czasowy")
plt.ylabel("Liczba zainfekowanych osób")
plt.title("Liczba zainfekowanych osób w zależności od typu")
plt.legend()
plt.grid(True)

plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.show()
