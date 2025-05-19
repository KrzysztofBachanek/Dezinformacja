import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import imageio.v2 as imageio
import os
import warnings

warnings.filterwarnings("ignore")

# PARAMETRY SYMULACJI
NUM_AGENTS = 2000                   # Liczba wszystkich agentów
NUM_INFLUENCERS = 3                 # Liczba influencerów
NUM_BOTS = 3                        # Liczba botów
NUM_SCEPTICS = 5                    # Liczba sceptyków
INFLUENCER_DEGREE = 5               # Minimalna liczba krawędzi influencera
BOT_DEGREE = 2                      # Minimalna liczba krawędzi bota

NUM_INITIAL_INFECTED = 2            # Liczba początkowo zakażonych
SIMULATION_STEPS = 20               # Liczba kroków czasowych
AWARENESS_START_STEP = 6            # Krok, od którego zaczyna się leczenie („awareness”)

FPS = 0.5                           # Klatki na sekundę w GIFie
GIF_FRAME_DURATION = 1000 * 1/FPS   # Długość trwania pojedyńczej klatki (w ms)
 

# KLASA AGENTA
class Agent:
    """
    Reprezentuje pojedynczego agenta w symulacji rozprzestrzeniania dezinformacji.
    Każdy agent ma unikalny identyfikator, typ (normal, influencer, bot, skeptic),
    oraz zestaw parametrów zachowania zależnych od typu:
    - podatność na infekcję (resistance_infected),
    - skłonność do infekowania innych (spreading_prob_infected),
    - podatność na leczenie (resistance_aware),
    - skłonność do leczenia innych (spreading_prob_aware).
    """

    def __init__(self, id, agent_type='normal'):
        self.id = id                  # Unikalny identyfikator agenta (zwykle numer wierzchołka grafu)
        self.agent_type = agent_type  # Typ agenta: 'normal', 'influencer', 'bot', 'skeptic'
        self.infected = False         # Czy agent jest obecnie zainfekowany dezinformacją
        self.recovered = False        # Czy agent jest odporny (po „leczeniu”)
        self.set_behavior()           # Ustawienie zachowania agenta na podstawie jego typu

    def set_behavior(self):
        """
        Ustawia parametry behawioralne agenta w zależności od jego typu.
        Różne typy mają różne skłonności do infekowania i leczenia.
        """

        if self.agent_type == 'normal':
            self.spreading_prob_infected = 0.3   # Normalny agent ma umiarkowaną szansę rozprzestrzeniania infekcji
            self.resistance_infected = 0.6      # Umiarkowana odporność na infekcję (40% szansy na zakażenie)
            self.spreading_prob_aware = 0.2      # Stosunkowo wysoka skłonność do leczenia innych po uleczeniu
            self.resistance_aware = 0.8         # Raczej podatny na leczenie (20% szansy na skuteczne uleczenie)

        elif self.agent_type == 'influencer':
            self.spreading_prob_infected = 0.4   # Influencer ma wysoką skłonność do rozprzestrzeniania dezinformacji
            self.resistance_infected = 0.3      # Bardzo niska odporność – łatwo się zaraża
            self.spreading_prob_aware = 0.05      # Rzadko leczy innych, nawet jeśli wyzdrowieje
            self.resistance_aware = 0.95         # Jest mało podatny na leczenie (trudno go uleczyć)

        elif self.agent_type == 'bot':
            self.spreading_prob_infected = 0.5   # Boty bardzo aktywnie rozprzestrzeniają dezinformację
            self.resistance_infected = 0.0       # Boty są w pełni podatne (zawsze się zarażą) – efekt wirusowej propagandy
            # Brak parametrów „awareness” – botów nie da się wyleczyć i nie leczą innych

        elif self.agent_type == 'skeptic':
            self.spreading_prob_infected = 0.15   # Sceptyk rzadko szerzy dezinformację – niski współczynnik infekcji
            self.resistance_infected = 0.7       # Bardzo odporny na zarażenie (tylko 30% szans na zainfekowanie)
            self.spreading_prob_aware = 0.5  # Bardzo chętnie i aktywnie leczy innych (świadomy zagrożenia)
            self.resistance_aware = 0.5          # Łatwo go uleczyć (50% szans na przyjęcie leczenia)


    def try_infect(self, neighbors, graph):
        """
        Próbuje zarazić sąsiadów z uwzględnieniem ich odporności oraz siły relacji (wagi połączenia).
        """
        for neighbor in neighbors:
            if not neighbor.infected and not neighbor.recovered:
                weight = graph[self.id][neighbor.id]['weight']
                effective_prob = self.spreading_prob_infected * weight
                if np.random.rand() < effective_prob and np.random.rand() > neighbor.resistance_infected:
                    neighbor.infected = True

    def try_cure_others(self, neighbors, graph):
        """
        Próbuje wyleczyć sąsiadów, uwzględniając siłę relacji oraz ich odporność na 'leczenie'.
        """
        if hasattr(self, 'spreading_prob_aware'):
            for neighbor in neighbors:
                if neighbor.agent_type in ['normal', 'skeptic', 'influencer']:
                    weight = graph[self.id][neighbor.id]['weight']
                    effective_prob = self.spreading_prob_aware * weight
                    if np.random.rand() < effective_prob and np.random.rand() > neighbor.resistance_aware:
                        neighbor.recovered = True
                        neighbor.infected = False


# TWORZENIE GRAFU I AGENTÓW
def create_custom_network(n, num_influencers, influencer_degree, num_bots, bot_degree, num_sceptics):
    G = nx.random_geometric_graph(n, 0.08)
    pos = nx.get_node_attributes(G, "pos")

    agents = {}
    all_ids = list(G.nodes)
    random.shuffle(all_ids)

    # Przydzielamy typy agentów
    influencer_ids = all_ids[:num_influencers]
    bot_ids = all_ids[num_influencers:num_influencers + num_bots]
    skeptic_ids = all_ids[num_influencers + num_bots:num_influencers + num_bots + num_sceptics]
    normal_ids = all_ids[num_influencers + num_bots + num_sceptics:]

    for i in influencer_ids:
        agents[i] = Agent(i, agent_type='influencer')
    for i in bot_ids:
        agents[i] = Agent(i, agent_type='bot')
    for i in skeptic_ids:
        agents[i] = Agent(i, agent_type='skeptic')
    for i in normal_ids:
        agents[i] = Agent(i, agent_type='normal')

    # Nadajemy losowe wagi istniejącym krawędziom (więzi 0.1–0.5)
    for u, v in G.edges():
        G[u][v]['weight'] = round(random.uniform(0.1, 0.5), 2)

    # Funkcja pomocnicza: dodaje losowe krawędzie o odpowiedniej wadze
    def add_edges_to_node(G, node, target_degree):
        while G.degree(node) < target_degree:
            potential_target = random.choice(list(G.nodes))
            if potential_target != node and not G.has_edge(node, potential_target):
                G.add_edge(node, potential_target, weight=round(random.uniform(0.1, 0.5), 2))
    
    # Wzmacniamy stopień węzłów influencerów i botów
    for node in influencer_ids:
        add_edges_to_node(G, node, influencer_degree)
    for node in bot_ids:
        add_edges_to_node(G, node, bot_degree)

    return G, agents, pos


# WIZUALIZACJA STANU GRAFU
def draw_graph(G, agents, pos, step):
    plt.figure(figsize=(30, 30))
    color_map = []

    for node in G.nodes():
        agent = agents[node]
        if agent.infected:
            color_map.append('red')
        elif agent.recovered:
            color_map.append('steelblue')
        elif agent.agent_type == 'bot':
            color_map.append('blue')
        elif agent.agent_type == 'influencer':
            color_map.append('yellow')
        elif agent.agent_type == 'skeptic':
            color_map.append('gray')
        else:
            color_map.append('green')

    nx.draw_networkx_edges(G, pos, alpha=0.2)
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color=color_map, edgecolors='black', alpha=0.9)

    plt.text(0.98, 0.98, f"Step {step}", fontsize=16, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.4'))

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.axis("off")

    os.makedirs("frames", exist_ok=True)
    filename = f"frames/frame_{step:03}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    return filename


# SYMULACJA ROZPRZESTRZENIANIA DEZINFORMACJI
def simulate_spread(G, agents, steps, initial_infected_count, pos):
    initial_infected = random.sample(list(agents.keys()), initial_infected_count)
    for i in initial_infected:
        agents[i].infected = True

    infected_over_time = []
    recovered_over_time = []
    frames = [draw_graph(G, agents, pos, 0)]

    for step in range(1, steps + 1):
        if step >= AWARENESS_START_STEP + 1 and np.random.rand() < 0.3:
            candidates = [a for a in agents.values() if a.agent_type in ['normal', 'skeptic'] and not a.recovered]
            if candidates:
                cured = random.choice(candidates)
                cured.recovered = True
                cured.infected = False
                cured.set_behavior()

        for node in G.nodes():
            if agents[node].infected:
                neighbors = [agents[n] for n in G.neighbors(node)]
                agents[node].try_infect(neighbors, G)

        for node in G.nodes():
            if agents[node].recovered:
                neighbors = [agents[n] for n in G.neighbors(node)]
                agents[node].try_cure_others(neighbors, G)


        infected_counts = {'normal': 0, 'influencer': 0, 'bot': 0, 'skeptic': 0}
        recovered_counts = {'normal': 0, 'influencer': 0, 'bot': 0, 'skeptic': 0}
        for a in agents.values():
            if a.infected:
                infected_counts[a.agent_type] += 1
            elif a.recovered:
                recovered_counts[a.agent_type] += 1

        infected_over_time.append(infected_counts)
        recovered_over_time.append(recovered_counts)
        frames.append(draw_graph(G, agents, pos, step))

    return infected_over_time, recovered_over_time, frames


# GENEROWANIE GIFA
def create_gif(frames, output_file="Dezinformacja - test.gif"):
    images = [imageio.imread(frame) for frame in frames]
    imageio.mimsave(output_file, images, format='GIF', duration=GIF_FRAME_DURATION)
    for frame in frames:
        os.remove(frame)
    os.rmdir("frames")


# URUCHOMIENIE PEŁNEJ SYMULACJI
G, agents, pos = create_custom_network(
    NUM_AGENTS, NUM_INFLUENCERS, INFLUENCER_DEGREE, NUM_BOTS, BOT_DEGREE, NUM_SCEPTICS
)
infected_data, recovered_data, frames = simulate_spread(
    G, agents, SIMULATION_STEPS, NUM_INITIAL_INFECTED, pos
)
create_gif(frames)
