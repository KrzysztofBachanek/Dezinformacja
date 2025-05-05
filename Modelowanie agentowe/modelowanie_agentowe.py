import threading
import numpy as np
import time

class Agent(threading.Thread):
    """
    Klasa reprezentująca pojedynczego agenta jako niezależny wątek.
    Agent może się poruszać, zarażać innych oraz być zarażonym. Może też pełnić rolę "influencera"
    z większym zasięgiem infekcji i ograniczoną mobilnością.

    Dziedziczenie po threading.Thread pozwala na uruchomienie każdego agenta jako osobny wątek.
    """

    def __init__(self, x, y, grid, lock, infected=False,
                 move_prob=0.2, infect_prob=0.8, is_influencer=False,
                 long_jump_prob=0.25, min_infect_radius=2, max_infect_radius=15):
        """
        Inicjalizacja agenta:
        - x, y: pozycja startowa na siatce
        - grid: odniesienie do wspólnej siatki (środowiska)
        - lock: blokada dla synchronizacji wątków
        - infected: czy agent jest początkowo zainfekowany
        - move_prob: prawdopodobieństwo ruchu w każdej iteracji
        - infect_prob: prawdopodobieństwo zarażenia innych agentów
        - is_influencer: czy agent jest tzw. influencerem (ma duży zasięg zarażania)
        - long_jump_prob: prawdopodobieństwo wykonania przeskoku (dalekiego ruchu)
        - min/max_infect_radius: zakres możliwego promienia infekcji dla influencerów
        """
        super().__init__()
        self.x = x
        self.y = y
        self.grid = grid
        self.lock = lock
        self.infected = infected
        self.move_prob = move_prob
        self.infect_prob = infect_prob
        self.running = True
        self.is_influencer = is_influencer
        self.long_jump_prob = long_jump_prob
        self.min_infect_radius = min_infect_radius
        self.max_infect_radius = max_infect_radius

        # Influencerzy mają większy losowy zasięg infekcji, pozostali agenci tylko 1
        self.infect_radius = np.random.randint(min_infect_radius, max_infect_radius + 1) if is_influencer else 1

    def run(self):
        """
        Główna pętla działania agenta:
        - Jeśli agent jest zainfekowany, próbuje zainfekować innych.
        - Agent może się poruszyć losowo, z szansą na "long jump".
        """
        while self.running:
            if self.infected:
                self.try_infect_neighbors()

            if not self.is_influencer and np.random.rand() < self.move_prob:
                if np.random.rand() < self.long_jump_prob:
                    self.long_jump()
                else:
                    self.try_move()

            time.sleep(0.002)  # Opóźnienie ograniczające zużycie CPU

    def try_move(self):
        """
        Próba wykonania standardowego ruchu w jedno z 8 sąsiednich pól.
        Ruch jest wykonywany tylko, jeśli nowe pole jest puste.
        """
        step_size = 3 if np.random.rand() < self.long_jump_prob else 1
        directions = [(-step_size, -step_size), (-step_size, 0), (-step_size, step_size),
                      (0, -step_size), (0, step_size),
                      (step_size, -step_size), (step_size, 0), (step_size, step_size)]
        np.random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[1]:
                with self.lock:
                    if self.grid[nx, ny] is None:
                        self.grid[self.x, self.y] = None
                        self.grid[nx, ny] = self
                        self.x, self.y = nx, ny
                        return

    def long_jump(self):
        """
        Wykonuje losowy przeskok w inne puste miejsce siatki — większy zasięg ruchu.
        """
        size = self.grid.shape[0]
        for _ in range(5):  # Maksymalnie 5 prób
            nx, ny = np.random.randint(0, size), np.random.randint(0, size)
            with self.lock:
                if self.grid[nx, ny] is None:
                    self.grid[self.x, self.y] = None
                    self.grid[nx, ny] = self
                    self.x, self.y = nx, ny
                    return

    def try_infect_neighbors(self):
        """
        Próba zainfekowania agentów znajdujących się w promieniu infekcji.
        Influencerzy mają większy promień.
        """
        r = self.infect_radius
        with self.lock:
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = self.x + dx, self.y + dy
                    if 0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[1]:
                        target = self.grid[nx, ny]
                        if target and not target.infected and np.random.rand() < self.infect_prob:
                            target.infected = True

    def stop(self):
        """
        Zatrzymuje pętlę działania agenta.
        """
        self.running = False

def init_grid(size, n_agents, n_infected_start, move_prob=0.6, infect_prob=0.9, n_influencers=5):
    """
    Inicjalizuje siatkę środowiska i tworzy agentów:
    - size: rozmiar siatki (size x size)
    - n_agents: liczba wszystkich agentów
    - n_infected_start: ilu z nich jest na starcie zainfekowanych
    - n_influencers: ilu z nich to influencerzy
    """
    grid = np.full((size, size), None)
    lock = threading.Lock()
    agents = []

    positions = set()
    while len(positions) < n_agents:
        x, y = np.random.randint(0, size, size=2)
        positions.add((x, y))

    for i, (x, y) in enumerate(positions):
        infected = i < n_infected_start
        is_influencer = i < n_influencers
        long_jump = 0.0 if is_influencer else 0.03
        agent = Agent(x, y, grid, lock, infected, move_prob, infect_prob, is_influencer, long_jump)
        grid[x, y] = agent
        agents.append(agent)

    return grid, agents, lock

def simulate(grid, agents, lock, steps=200):
    """
    Główna pętla symulacyjna — zbiera dane o pozycjach agentów w kolejnych krokach.
    - Zwraca listę klatek z pozycjami, kolorami i zasięgami (dla influencerów).
    """
    frames = []
    for step in range(steps):
        with lock:
            data = [(a.x, a.y, 'black' if a.is_influencer else 'red' if a.infected else 'green', a.infect_radius) for a in agents]
        frames.append(data)
        time.sleep(0.01)
    return frames


# Funkcja do tworzenia GIF-a na podstawie klatek symulacji

import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image

def makeGif(frames, name="SYMULACJA.gif", size=100):
    """
    Tworzy animowany GIF przedstawiający przebieg symulacji:
    - Kolor czerwony: zainfekowani
    - Kolor zielony: zdrowi
    - Kolor czarny: influencerzy (z animowanym zasięgiem infekcji)
    """
    images = []

    for i, agents_data in enumerate(frames):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")

        # Nadrzędny tytuł pokazujący czas i liczbę zainfekowanych
        plt.suptitle(f"Czas: {i} | Liczba zarażonych: {sum(1 for _, _, c, _ in agents_data if c == 'red')}", fontsize=12)

        # Czarna ramka wokół siatki
        ax.plot([0, size, size, 0, 0], [0, 0, size, size, 0], color='black', linewidth=1)

        for x, y, color, radius in agents_data:
            ax.scatter(x, y, c=color, s=12)
            if color == 'black':
                # Pulsujący promień zasięgu infekcji influencera
                pulse_radius = radius * (1 + 0.2 * np.sin(i * 0.3))
                circle = plt.Circle((x, y), pulse_radius, color='purple', alpha=0.25)
                ax.add_patch(circle)

        # Renderowanie wykresu jako obrazek
        canvas = FigureCanvas(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        img = Image.open(buf)
        images.append(img)
        plt.close(fig)

    # Tworzenie animacji (loop=1 oznacza brak zapętlenia)
    images[0].save(name, save_all=True, append_images=images[1:], duration=80, loop=1)


# URUCHOMIENIE SYMULACJI

# Parametry symulacji
size = 200
n_agents = 200
n_infected_start = 10
steps = 400

# Tworzenie środowiska
grid, agents, lock = init_grid(size, n_agents, n_infected_start, move_prob=0.8, infect_prob=0.4)

# Start wątków agentów
for agent in agents:
    agent.start()

# Zbieranie danych do animacji
frames = simulate(grid, agents, lock, steps=steps+1)

# Zatrzymanie działania agentów i dołączenie wątków
for agent in agents:
    agent.stop()
    agent.join()

# Generowanie animowanego GIF-a z przebiegiem symulacji
makeGif(frames, name="SYMULACJA.gif", size=size)
