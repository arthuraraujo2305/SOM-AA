import numpy as np


class GrowingSOM:
    """
    Implementação prática de um Growing SOM para usar na STM.

    Ideia:
    - começa com grade 2x2
    - encontra BMU
    - acumula erro no BMU
    - atualiza BMU e vizinhos com gaussiana
    - se erro do BMU passar do growth_threshold, cresce ao redor dele
    """

    def __init__(
        self,
        input_dim: int,
        growth_threshold: float = 1.0,
        spread_factor: float = 0.9,
        learning_rate: float = 0.5,
        sigma: float = 1.0,
        max_nodes: int = 100,
        random_seed: int = 10,
    ):
        self.input_dim = input_dim
        self.growth_threshold = growth_threshold
        self.spread_factor = spread_factor
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.max_nodes = max_nodes
        self.random_state = np.random.RandomState(random_seed)

        # dicionário: posição -> vetor de pesos
        self.nodes = {}
        self.errors = {}
        self.hits = {}

        # inicialização 2x2
        initial_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for pos in initial_positions:
            self.nodes[pos] = np.zeros(self.input_dim, dtype=float)
            self.errors[pos] = 0.0
            self.hits[pos] = 0

    def initialize_from_data(self, data: np.ndarray):
        """
        Inicializa os pesos dos 4 neurônios com amostras da STM.
        """
        n = len(data)
        if n == 0:
            return

        positions = list(self.nodes.keys())
        for i, pos in enumerate(positions):
            idx = self.random_state.randint(0, n)
            self.nodes[pos] = data[idx].copy()

    def _grid_distance(self, pos_a, pos_b) -> float:
        return np.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)

    def _find_bmu(self, x: np.ndarray):
        best_pos = None
        best_dist = float("inf")

        for pos, w in self.nodes.items():
            d = np.linalg.norm(x - w)
            if d < best_dist:
                best_dist = d
                best_pos = pos

        return best_pos, best_dist

    def _gaussian_neighborhood(self, grid_dist: float) -> float:
        if self.sigma <= 0:
            return 0.0
        return np.exp(-(grid_dist ** 2) / (2 * (self.sigma ** 2)))

    def _neighbors_4(self, pos):
        r, c = pos
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]

    def _insert_new_node(self, new_pos, ref_pos):
        """
        Insere novo neurônio perto do neurônio de referência.
        Peso inicial:
        - média entre ref_pos e vizinhos existentes, se houver
        - senão, cópia do peso do ref_pos
        """
        if new_pos in self.nodes:
            return

        if len(self.nodes) >= self.max_nodes:
            return

        ref_w = self.nodes[ref_pos]
        valid_neighbor_weights = []

        for nb in self._neighbors_4(new_pos):
            if nb in self.nodes:
                valid_neighbor_weights.append(self.nodes[nb])

        if len(valid_neighbor_weights) > 0:
            new_w = np.mean(np.vstack([ref_w] + valid_neighbor_weights), axis=0)
        else:
            new_w = ref_w.copy()

        self.nodes[new_pos] = new_w
        self.errors[new_pos] = self.errors[ref_pos] * self.spread_factor
        self.hits[new_pos] = 0

    def _grow_around_bmu(self, bmu_pos):
        """
        Regra simples de crescimento:
        tenta criar neurônios nas 4 posições adjacentes ao BMU.
        """
        for new_pos in self._neighbors_4(bmu_pos):
            if len(self.nodes) >= self.max_nodes:
                break
            self._insert_new_node(new_pos, bmu_pos)

        # reduz erro do BMU após crescimento
        self.errors[bmu_pos] *= self.spread_factor

    def train(self, data: np.ndarray, num_epochs: int = 10):
        if len(data) == 0:
            return

        self.initialize_from_data(data)

        for epoch in range(num_epochs):
            for x in data:
                bmu_pos, bmu_dist = self._find_bmu(x)

                # acumula erro do BMU
                self.errors[bmu_pos] += bmu_dist
                self.hits[bmu_pos] += 1

                # atualiza BMU e vizinhos com gaussiana
                for pos in list(self.nodes.keys()):
                    grid_dist = self._grid_distance(pos, bmu_pos)
                    h = self._gaussian_neighborhood(grid_dist)

                    if h <= 1e-8:
                        continue

                    self.nodes[pos] += self.learning_rate * h * (x - self.nodes[pos])

                # crescimento
                if self.errors[bmu_pos] > self.growth_threshold and len(self.nodes) < self.max_nodes:
                    self._grow_around_bmu(bmu_pos)

    def get_cluster_labels(self, data: np.ndarray):
        labels = []
        for x in data:
            bmu_pos, _ = self._find_bmu(x)
            labels.append((int(bmu_pos[0]), int(bmu_pos[1])))
        return labels

    def get_node_weights(self):
        return self.nodes

    def get_active_nodes(self):
        return [(int(pos[0]), int(pos[1])) for pos, count in self.hits.items() if count > 0]