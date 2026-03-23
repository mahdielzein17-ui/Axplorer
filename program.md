# program

This is the workflow for implementing a new mathematical problem in Axplorer.

## Phase 1: Problem Implementation

The user describes a new discrete optimization problem. You implement it as an Axplorer environment.

### Understanding the problem

Before writing any code, make sure you understand:

1. **What is the object?** (graph, point set, matrix, sequence, etc.)
2. **What is the parameter N?** (number of nodes, grid size, sequence length, etc.)
3. **What are the constraints?** (forbidden substructures, required properties, etc.)
4. **What is the score?** (what to maximize — number of edges, number of points, etc.)
5. **How to tokenize the object?** 

Ask the user to clarify anything ambiguous. Do not guess.

### Writing the environment

Read the existing environments for reference:

- `envs/environment.py` — base classes `DataPoint` and `BaseEnvironment`
- `envs/cycle.py` — example: 4-cycle-free graphs (Turan problem)
- `envs/isosceles.py` — example: isosceles-free point sets
- `envs/sphere.py` — example: 5-cospherical-free point sets
- `envs/tokenizers.py` — available tokenizers
- `new_envs.ipynb` — tutorial notebook for creating new environments

Create `envs/<problem_name>.py` with:

1. **`<Problem>DataPoint(DataPoint)`** class:
   - `__init__(self, N, init=False)` — if `init=True`, generate a random valid instance
   - `calc_score(self)` — return the objective value (>=0 if valid, -1 if invalid). **This is the most critical method.** It must be correct, efficient, and match the mathematical definition exactly.
   - `calc_features(self)` — return a canonical string representation for deduplication
   - `local_search(self)` — repair invalid solutions and optionally improve valid ones. Typically: (a) remove elements that cause constraint violations, (b) greedily add elements that don't violate constraints.

2. **`<Problem>Environment(BaseEnvironment)`** class:
   - Set `k`, `are_coordinates_symmetric`, `data_class`
   - Choose tokenizer
   - Implement `register_args(parser)` if the problem has extra parameters

3. **Register** in `envs/__init__.py` by adding the new environment to `ENVS`.

### Implementation guidelines

- Use **numba** (`@njit`) for inner loops in scoring and local search when performance matters.
- Keep `calc_score()` **simple and obviously correct** — this is the function the user will audit. Prefer clarity over cleverness.
- `local_search()` should be greedy and fast. It doesn't need to be optimal — the transformer + iteration loop will handle exploration.
- For random generation in `__init__`, start with something simple (e.g. random subset, random permutation) even if the results are low quality. The training loop improves from there.
- Follow the patterns in existing environments. Don't reinvent conventions.

## Phase 2: User Review

**STOP HERE.** Present the implementation to the user for review before running anything.

Specifically ask the user to verify:

1. **`calc_score()`** — Is the scoring function mathematically correct? Does it match the problem definition? Are edge cases handled?
2. **Constraint checking** — Are all forbidden configurations detected?
3. **`local_search()`** — Does the repair strategy make sense for this problem?
4. **Tokenization choice** — Is the encoding appropriate for the problem structure?

The user may also provide:

- **Mathematical insights**: known bounds, constructions, symmetries, or structural properties that could inform the local search or generation strategy.
- **Bug fixes**: corrections to the scoring logic or constraint detection.
- **Performance suggestions**: better algorithms for constraint checking.

Incorporate all feedback before proceeding. If the user provides known optimal constructions or bounds, note them — they serve as targets for training.

## Phase 3: Test

Verify the implementation works end-to-end:

1. **Unit test the scoring function** by running a quick sanity check in Python:
   ```bash
   python -c "
   from src.envs.<problem_name> import <Problem>DataPoint
   # Generate a few random instances
   for _ in range(10):
       dp = <Problem>DataPoint(N=<small_N>, init=True)
       dp.calc_score()
       print(f'score={dp.score}')
   "
   ```

2. **Run a minimal training pass** with small parameters to verify the full pipeline:
   ```bash
   python train.py \
       --env_name <problem_name> \
       --exp_name smoke_test \
       --N <small_N> \
       --max_epochs 2 \
       --max_steps 100 \
       --gensize 1000 \
       --num_samples_from_model 500 \
       --pop_size 500 \
       --batch_size 16 \
       --n_layer 2 \
       --n_embd 128
   ```

3. Check that:
   - No crashes or errors
   - Scores are being computed and logged
   - The model samples produce decodable outputs
   - Local search runs without errors

If the smoke test fails, fix the issue and re-run. 