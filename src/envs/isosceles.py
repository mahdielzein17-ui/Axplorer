import numpy as np
from numba import njit

from src.envs.environment import BaseEnvironment, DataPoint
from src.envs.tokenizers import DenseTokenizer, SparseTokenizerSequenceKTokens, SparseTokenizerSingleInteger
from src.envs.utils import canonical_form_2d_symmetric, random_symmetry_2d_symmetric
from src.utils import bool_flag


@njit(cache=True)
def _greedy_fill_jittered(points_arr, n_points):
    if n_points < 3:
        return np.empty((0, 6), dtype=np.int32)

    max_triangles = n_points * (n_points - 1) * (n_points - 2) // 2
    triangles = np.empty((max_triangles, 6), dtype=np.int32)
    idx = 0

    for i in range(n_points):
        ax, ay = points_arr[i, 0], points_arr[i, 1]

        n_others = n_points - 1
        if n_others < 2:
            continue

        distances = np.empty(n_others, dtype=np.int64)
        indices = np.empty(n_others, dtype=np.int32)

        k = 0
        for j in range(n_points):
            if j == i:
                continue
            bx, by = points_arr[j, 0], points_arr[j, 1]
            dx = ax - bx
            dy = ay - by
            distances[k] = dx * dx + dy * dy
            indices[k] = j
            k += 1

        order = np.argsort(distances)

        start = 0
        while start < n_others:
            d2 = distances[order[start]]
            end = start + 1
            while end < n_others and distances[order[end]] == d2:
                end += 1

            if end - start >= 2:
                for p in range(start, end):
                    for q in range(p + 1, end):
                        j1 = indices[order[p]]
                        j2 = indices[order[q]]
                        triangles[idx, 0] = ax
                        triangles[idx, 1] = ay
                        triangles[idx, 2] = points_arr[j1, 0]
                        triangles[idx, 3] = points_arr[j1, 1]
                        triangles[idx, 4] = points_arr[j2, 0]
                        triangles[idx, 5] = points_arr[j2, 1]
                        idx += 1

            start = end

    return triangles[:idx]


@njit(cache=True)
def _has_isosceles_conflict(points_arr, n_points, new_x, new_y):
    if n_points < 2:
        return False

    new_distances = np.empty(n_points, dtype=np.int64)
    for i in range(n_points):
        dx = new_x - points_arr[i, 0]
        dy = new_y - points_arr[i, 1]
        new_distances[i] = dx * dx + dy * dy

    for i in range(n_points):
        for j in range(i + 1, n_points):
            if new_distances[i] == new_distances[j]:
                return True

    for i in range(n_points):
        d_new = new_distances[i]
        for j in range(n_points):
            if i == j:
                continue
            dx = points_arr[i, 0] - points_arr[j, 0]
            dy = points_arr[i, 1] - points_arr[j, 1]
            if dx * dx + dy * dy == d_new:
                return True

    return False


@njit(cache=True)
def _data_to_expanded_grid(data, expanded_grid, N):
    for x in range(N):
        for y in range(N):
            if data[x, y] == 1:
                expanded_grid[x, y] = 1
                expanded_grid[2 * N - 1 - x, y] = 1
                expanded_grid[x, 2 * N - 1 - y] = 1
                expanded_grid[2 * N - 1 - x, 2 * N - 1 - y] = 1


@njit(cache=True)
def _greedy_add_symmetric(data, expanded_grid, candidates, N):
    max_points = 4 * N * N
    points_arr = np.empty((max_points, 2), dtype=np.int32)
    n_points = 0

    for x in range(2 * N):
        for y in range(2 * N):
            if expanded_grid[x, y] == 1:
                points_arr[n_points, 0] = x
                points_arr[n_points, 1] = y
                n_points += 1

    for enc in candidates:
        x, y = enc // N, enc % N
        if data[x, y] == 1:
            continue

        sym_points = [(x, y), (2 * N - 1 - x, y), (x, 2 * N - 1 - y), (2 * N - 1 - x, 2 * N - 1 - y)]

        # Temporarily add the 3 symmetric points to points_arr
        # no need to remove if it doesn't work because it will rewrite from the next point
        for i in range(3):
            sx, sy = sym_points[i + 1]
            points_arr[n_points + i, 0] = sx
            points_arr[n_points + i, 1] = sy

        if not _has_isosceles_conflict(points_arr, n_points + 3, x, y):
            data[x, y] = 1
            for sx, sy in sym_points:
                expanded_grid[sx, sy] = 1
            points_arr[n_points + 3, 0] = x
            points_arr[n_points + 3, 1] = y
            n_points += 4


@njit(cache=True)
def _greedy_remove_symmetric(data, expanded_grid, triangles, N):
    num_triangles = len(triangles)
    if num_triangles == 0:
        return

    active = np.ones(num_triangles, dtype=np.uint8)

    point_count = np.zeros((2 * N, 2 * N), dtype=np.int32)
    for t in range(num_triangles):
        point_count[triangles[t, 0], triangles[t, 1]] += 1
        point_count[triangles[t, 2], triangles[t, 3]] += 1
        point_count[triangles[t, 4], triangles[t, 5]] += 1

    num_active = num_triangles

    while num_active > 0:

        max_count = 0
        best_x, best_y = -1, -1
        for x in range(N):
            for y in range(N):
                if data[x, y] == 1:
                    total_count = (
                        point_count[x, y] + point_count[2 * N - 1 - x, y] + point_count[x, 2 * N - 1 - y] + point_count[2 * N - 1 - x, 2 * N - 1 - y]
                    )
                    if total_count > max_count:
                        max_count = total_count
                        best_x, best_y = x, y

        if max_count == 0:
            break

        data[best_x, best_y] = 0

        sym_points = [(best_x, best_y), (2 * N - 1 - best_x, best_y), (best_x, 2 * N - 1 - best_y), (2 * N - 1 - best_x, 2 * N - 1 - best_y)]

        for sx, sy in sym_points:
            expanded_grid[sx, sy] = 0

        for t in range(num_triangles):
            if not active[t]:
                continue

            contains = False
            for sx, sy in sym_points:
                if (
                    (triangles[t, 0] == sx and triangles[t, 1] == sy)
                    or (triangles[t, 2] == sx and triangles[t, 3] == sy)
                    or (triangles[t, 4] == sx and triangles[t, 5] == sy)
                ):
                    contains = True
                    break

            if contains:
                active[t] = False
                num_active -= 1
                point_count[triangles[t, 0], triangles[t, 1]] -= 1
                point_count[triangles[t, 2], triangles[t, 3]] -= 1
                point_count[triangles[t, 4], triangles[t, 5]] -= 1


class IsoscelesDataPoint(DataPoint):
    MAKE_OBJECT_CANONICAL = False

    def __init__(self, N, init=False):
        super().__init__()
        self.N = N
        self.data = np.zeros((self.N, self.N), dtype=np.uint8)
        self.expanded_grid = np.zeros((2 * self.N, 2 * self.N), dtype=np.uint8)
        self.isosceles = np.empty((0, 6), dtype=np.int32)
        if init:
            self._add_points_greedily()
            if self.MAKE_OBJECT_CANONICAL:
                self.data = canonical_form_2d_symmetric(self.data)
                self._sync_expanded_grid()
            self.calc_features()
            self.calc_score()

    def _sync_expanded_grid(self):
        self.expanded_grid.fill(0)
        _data_to_expanded_grid(self.data, self.expanded_grid, self.N)

    def calc_score(self):
        if self.isosceles.size > 0:
            self.score = -1
        else:
            self.score = self.expanded_grid.sum().item()

    def calc_features(self):
        w = []
        for i in range(self.N):
            for j in range(self.N):
                w.append(self.data[i, j])
        self.features = ",".join(map(str, w))

    def _add_points_greedily(self):
        np.random.seed(None)
        candidates = np.arange(self.N * self.N, dtype=np.int32)
        np.random.shuffle(candidates)
        _greedy_add_symmetric(self.data, self.expanded_grid, candidates, self.N)

    def _remove_points_greedily(self):
        if self.isosceles.size > 0:
            _greedy_remove_symmetric(self.data, self.expanded_grid, self.isosceles, self.N)
            self.isosceles = np.empty((0, 6), dtype=np.int32)

    def _isosceles_computation(self):
        points = np.argwhere(self.expanded_grid == 1)
        points_arr = np.ascontiguousarray(points, dtype=np.int32)
        self.isosceles = _greedy_fill_jittered(points_arr, len(points_arr))

    def local_search(self, improve_with_local_search):
        self._isosceles_computation()
        self._remove_points_greedily()
        if improve_with_local_search:
            self._add_points_greedily()
        self._isosceles_computation()
        self.calc_score()
        if self.MAKE_OBJECT_CANONICAL:
            self.data = canonical_form_2d_symmetric(self.data)
            self._sync_expanded_grid()
        self.calc_features()

    @classmethod
    def _update_class_params(cls, pars):
        cls.MAKE_OBJECT_CANONICAL = pars

    @classmethod
    def _save_class_params(cls):
        return cls.MAKE_OBJECT_CANONICAL


class IsoscelesEnvironment(BaseEnvironment):
    # this problem lives in N^2, therefore k=2
    # (i, j) or (j, i) represents two distinct points in the grid, therefore are_coordinates_symmetric=False
    k = 2
    are_coordinates_symmetric = False
    data_class = IsoscelesDataPoint

    def __init__(self, params):
        super().__init__(params)
        self.data_class.MAKE_OBJECT_CANONICAL = params.make_object_canonical
        encoding_augmentation = random_symmetry_2d_symmetric if params.augment_data_representation else None
        if params.encoding_tokens == "single_integer":
            self.tokenizer = SparseTokenizerSingleInteger(
                self.data_class, params.N, self.k, self.are_coordinates_symmetric, self.SPECIAL_SYMBOLS, encoding_augmentation=encoding_augmentation
            )
        elif params.encoding_tokens == "sequence_k_tokens":
            self.tokenizer = SparseTokenizerSequenceKTokens(
                self.data_class, params.N, self.k, self.are_coordinates_symmetric, self.SPECIAL_SYMBOLS, encoding_augmentation=encoding_augmentation
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
        """
        Register environment parameters.
        """
        parser.add_argument("--N", type=int, default=30, help="Half grid size N. Total grid size is 2N")
        parser.add_argument("--encoding_tokens", type=str, default="single_integer", help="single_integer/sequence_k_tokens/adjacency")
        parser.add_argument("--make_object_canonical", type=bool_flag, default="false", help="sort the grid by symmetry")
        parser.add_argument(
            "--augment_data_representation", type=bool_flag, default="false", help="augment the data representation with predefined function"
        )
        parser.add_argument("--pow2base", type=int, default=1, help="Number of adjacency entries to code together")
