import numpy as np

from src.envs.environment import BaseEnvironment, DataPoint
from src.envs.tokenizers import (
    DenseTokenizer,
    SparseTokenizerSequenceKTokens,
    SparseTokenizerSingleInteger,
)
from src.envs.utils import random_symmetry_adj_matrix, sort_graph_based_on_degree
from src.utils import bool_flag


class HexagonDataPoint(DataPoint):
    """
    Represents a C6-free graph on N vertices.
    Goal: maximize edges while avoiding 6-cycles.
    """

    MAKE_OBJECT_CANONICAL = False

    def __init__(self, N, init=False):
        super().__init__()
        self.N = N
        self.data = np.zeros((self.N, self.N), dtype=np.uint8)
        self.cycles = []

        if init:
            self._add_edges_greedily()
            if self.MAKE_OBJECT_CANONICAL:
                self.data = sort_graph_based_on_degree(self.data)
            self.calc_features()
            self.calc_score()

    def calc_score(self):
        if len(self.cycles) > 0:
            self.score = -1
        else:
            self.score = self.data.sum().item() // 2

    def calc_features(self):
        w = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                w.append(self.data[i, j])
        self.features = ",".join(map(str, w))

    def _has_c6_through_edge(self, u, v):
        """
        Check if there's a 6-cycle containing edge (u, v).
        A 6-cycle through (u,v) means there's a path of length 5
        from u to v not using the direct edge (u,v).

        We enumerate: u - a - b - c - d - v where all vertices are distinct
        and all edges exist, and {a,b,c,d} are distinct from u,v.
        """
        A = self.data
        N = self.N
        # neighbors of u and v (excluding each other)
        nbrs_u = [i for i in range(N) if A[u, i] == 1 and i != v]
        nbrs_v = [i for i in range(N) if A[v, i] == 1 and i != u]

        # path: u - a - b - c - d - v
        for a in nbrs_u:
            for b in range(N):
                if A[a, b] == 0 or b == u or b == a:
                    continue
                for c in range(N):
                    if A[b, c] == 0 or c == u or c == a or c == b:
                        continue
                    for d in nbrs_v:
                        if d == u or d == a or d == b or d == c:
                            continue
                        if A[c, d] == 1:
                            return True
        return False

    def _add_edges_greedily(self):
        """
        Greedily add edges while avoiding 6-cycles.
        For efficiency, we maintain a quick check before doing the full C6 test.
        """
        np.random.seed(None)
        allowed_edges = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.data[i, j] == 0:
                    allowed_edges.append((i, j))
        np.random.shuffle(allowed_edges)

        for i, j in allowed_edges:
            if self.data[i, j] == 1:
                continue
            # tentatively add
            self.data[i, j] = 1
            self.data[j, i] = 1
            if self._has_c6_through_edge(i, j):
                # undo
                self.data[i, j] = 0
                self.data[j, i] = 0

    def _cycles_computation(self):
        """
        Find all 6-cycles in the graph.
        A 6-cycle is (v0, v1, v2, v3, v4, v5) where each consecutive pair
        (and v5-v0) are edges, and all vertices are distinct.

        We enumerate by fixing the lexicographically smallest vertex,
        then finding paths of length 5 back to it.
        """
        A = self.data
        N = self.N
        cycles_set = set()

        # build adjacency lists for speed
        adj = [[] for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if A[i, j] == 1:
                    adj[i].append(j)
                    adj[j].append(i)

        # for each edge (v0, v1), find paths v1 - v2 - v3 - v4 - v5 - v0
        # where v0 < all other vertices in the cycle (to avoid counting each cycle 12 times)
        for v0 in range(N):
            for v1 in adj[v0]:
                if v1 <= v0:
                    continue
                for v2 in adj[v1]:
                    if v2 <= v0 or v2 == v0:
                        continue
                    for v3 in adj[v2]:
                        if v3 <= v0 or v3 == v1 or v3 == v0:
                            continue
                        for v4 in adj[v3]:
                            if v4 <= v0 or v4 == v2 or v4 == v1 or v4 == v0:
                                continue
                            for v5 in adj[v4]:
                                if v5 <= v0 or v5 == v3 or v5 == v2 or v5 == v1:
                                    continue
                                if A[v5, v0] == 1:
                                    # found a 6-cycle: v0-v1-v2-v3-v4-v5-v0
                                    # canonicalize: take the smallest rotation/reflection
                                    ring = [v0, v1, v2, v3, v4, v5]
                                    # since v0 is smallest, just pick direction
                                    if ring[1] < ring[5]:
                                        canon = tuple(ring)
                                    else:
                                        canon = (
                                            ring[0],
                                            ring[5],
                                            ring[4],
                                            ring[3],
                                            ring[2],
                                            ring[1],
                                        )
                                    cycles_set.add(canon)

        # convert to edge-based representation for removal
        self.cycles = []
        for cycle in cycles_set:
            edges = []
            for k in range(6):
                a, b = cycle[k], cycle[(k + 1) % 6]
                edges.append((min(a, b), max(a, b)))
            self.cycles.append(tuple(edges))

    def _remove_edges_greedily(self):
        """
        Greedily remove edges to eliminate all 6-cycles.
        Remove the edge appearing in the most cycles.
        """
        while self.cycles:
            edge_count = {}
            for cycle in self.cycles:
                for edge in cycle:
                    edge_count[edge] = edge_count.get(edge, 0) + 1
            i, j = max(edge_count, key=edge_count.get)
            self.data[i, j] = 0
            self.data[j, i] = 0

            remaining_cycles = []
            for cycle in self.cycles:
                if (i, j) not in cycle:
                    remaining_cycles.append(cycle)
            self.cycles = remaining_cycles

    def local_search(self, improve_with_local_search):
        self._cycles_computation()
        self._remove_edges_greedily()
        if improve_with_local_search:
            self._add_edges_greedily()
        self.cycles = []
        if self.MAKE_OBJECT_CANONICAL:
            self.data = sort_graph_based_on_degree(self.data)
        self.calc_features()
        self.calc_score()

    @classmethod
    def _update_class_params(cls, pars):
        cls.MAKE_OBJECT_CANONICAL = pars

    @classmethod
    def _save_class_params(cls):
        return cls.MAKE_OBJECT_CANONICAL


class HexagonEnvironment(BaseEnvironment):
    k = 2
    are_coordinates_symmetric = True
    data_class = HexagonDataPoint

    def __init__(self, params):
        super().__init__(params)
        self.data_class.MAKE_OBJECT_CANONICAL = params.make_object_canonical
        encoding_augmentation = (
            random_symmetry_adj_matrix if params.augment_data_representation else None
        )
        if params.encoding_tokens == "single_integer":
            self.tokenizer = SparseTokenizerSingleInteger(
                self.data_class,
                params.N,
                self.k,
                self.are_coordinates_symmetric,
                self.SPECIAL_SYMBOLS,
                encoding_augmentation=encoding_augmentation,
            )
        elif params.encoding_tokens == "sequence_k_tokens":
            self.tokenizer = SparseTokenizerSequenceKTokens(
                self.data_class,
                params.N,
                self.k,
                self.are_coordinates_symmetric,
                self.SPECIAL_SYMBOLS,
                encoding_augmentation=encoding_augmentation,
            )
        elif params.encoding_tokens == "adjacency":
            self.tokenizer = DenseTokenizer(
                self.data_class,
                params.N,
                self.k,
                self.are_coordinates_symmetric,
                self.SPECIAL_SYMBOLS,
                pow2base=params.pow2base,
                encoding_augmentation=encoding_augmentation,
            )
        else:
            raise ValueError(f"Invalid encoding: {params.encoding_tokens}")

    @staticmethod
    def register_args(parser):
        parser.add_argument(
            "--N", type=int, default=30, help="Number of vertices in the C6-free graph"
        )
        parser.add_argument(
            "--encoding_tokens",
            type=str,
            default="single_integer",
            help="single_integer/sequence_k_tokens/adjacency",
        )
        parser.add_argument(
            "--make_object_canonical",
            type=bool_flag,
            default="false",
            help="sort the graph node names based on its indegree",
        )
        parser.add_argument(
            "--augment_data_representation",
            type=bool_flag,
            default="false",
            help="augment the data representation with predefined function",
        )
        parser.add_argument(
            "--pow2base",
            type=int,
            default=1,
            help="Number of adjacency entries to code together",
        )
