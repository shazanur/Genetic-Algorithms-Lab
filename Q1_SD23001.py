#Q1
import math
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# -------------------- Problem Definitions --------------------
@dataclass
class GAProblem:
    name: str
    chromosome_type: str  
    dim: int
    bounds: Tuple[float, float] | None
    fitness_fn: Callable[[np.ndarray], float]


def make_peak_40(dim: int = 80, target_ones: int = 40, max_fitness: float = 80.0) -> GAProblem:
    """
    Fitness peaks at target_ones and has max_fitness at the peak.
    Triangular landscape:
        fitness = max_fitness - 2 * abs(ones - target_ones)

    For dim=80, target_ones=40:
        abs(ones-40) ranges 0..40
        fitness ranges 80..0
    """
    def fitness(x: np.ndarray) -> float:
        ones = int(np.sum(x))
        return float(max_fitness - 2.0 * abs(ones - target_ones))

    return GAProblem(
        name=f"Peak Fitness at Ones={target_ones} ({dim} bits)",
        chromosome_type="bit",
        dim=dim,
        bounds=None,
        fitness_fn=fitness,
    )


# -------------------- GA Operators --------------------
def init_population(problem: GAProblem, pop_size: int, rng: np.random.Generator) -> np.ndarray:
    # Only bit chromosomes are used for this requirement
    return rng.integers(0, 2, size=(pop_size, problem.dim), dtype=np.int8)


def tournament_selection(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    idxs = rng.integers(0, fitness.size, size=k)
    best = idxs[np.argmax(fitness[idxs])]
    return int(best)


def one_point_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if a.size <= 1:
        return a.copy(), b.copy()
    point = int(rng.integers(1, a.size))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2


def bit_mutation(x: np.ndarray, mut_rate: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(x.shape) < mut_rate
    y = x.copy()
    y[mask] = 1 - y[mask]
    return y


def evaluate(pop: np.ndarray, problem: GAProblem) -> np.ndarray:
    return np.array([problem.fitness_fn(ind) for ind in pop], dtype=float)


def run_ga(
    problem: GAProblem,
    pop_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    tournament_k: int,
    elitism: int,
    seed: int | None,
    stream_live: bool = True,
):
    rng = np.random.default_rng(seed)
    pop = init_population(problem, pop_size, rng)
    fit = evaluate(pop, problem)

    # Live UI containers
    chart_area = st.empty()
    best_area = st.empty()

    history_best: List[float] = []
    history_avg: List[float] = []
    history_worst: List[float] = []

    for gen in range(generations):
        # Logging
        best_idx = int(np.argmax(fit))
        best_fit = float(fit[best_idx])
        avg_fit = float(np.mean(fit))
        worst_fit = float(np.min(fit))
        history_best.append(best_fit)
        history_avg.append(avg_fit)
        history_worst.append(worst_fit)

        # Live updates
        if stream_live:
            df_live = pd.DataFrame(
                {"Best": history_best, "Average": history_avg, "Worst": history_worst}
            )
            chart_area.line_chart(df_live)
            best_area.markdown(
                f"Generation {gen+1}/{generations} — Best fitness: **{best_fit:.2f}**"
            )

        # Elitism: keep top E
        E = max(0, min(elitism, pop_size))
        elite_idx = np.argpartition(fit, -E)[-E:] if E > 0 else np.array([], dtype=int)
        elites = pop[elite_idx].copy() if E > 0 else np.empty((0, problem.dim), dtype=np.int8)

        # Create next generation
        next_pop: List[np.ndarray] = []
        while len(next_pop) < pop_size - E:
            # Select parents
            i1 = tournament_selection(fit, tournament_k, rng)
            i2 = tournament_selection(fit, tournament_k, rng)
            p1, p2 = pop[i1], pop[i2]

            # Crossover
            if rng.random() < crossover_rate:
                c1, c2 = one_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            c1 = bit_mutation(c1, mutation_rate, rng)
            c2 = bit_mutation(c2, mutation_rate, rng)

            next_pop.append(c1)
            if len(next_pop) < pop_size - E:
                next_pop.append(c2)

        # Insert elites and finalize
        pop = np.vstack([np.array(next_pop, dtype=np.int8), elites]) if E > 0 else np.array(next_pop, dtype=np.int8)
        fit = evaluate(pop, problem)

    # Final metrics and best solution
    best_idx = int(np.argmax(fit))
    best = pop[best_idx].copy()
    best_fit = float(fit[best_idx])

    history_df = pd.DataFrame({"Best": history_best, "Average": history_avg, "Worst": history_worst})

    return {
        "best": best,
        "best_fitness": best_fit,
        "history": history_df,
        "final_population": pop,
        "final_fitness": fit,
    }


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Genetic Algorithm - Bit Pattern Generator", page_icon="🧬", layout="wide")
st.title("Genetic Algorithm (GA) — Bit Pattern Generator")
st.caption("Generates a bit pattern using fixed GA parameters (as required).")

# ---- Fixed requirements ----
POP_SIZE = 300
CHROM_LEN = 80
GENERATIONS = 50
TARGET_ONES = 40
MAX_FITNESS = 80.0

CROSSOVER_RATE = 0.90
MUTATION_RATE = 1.0 / CHROM_LEN  
TOURNAMENT_K = 3
ELITISM = 2

with st.sidebar:
    st.header("Fixed Requirements (Locked)")
    st.markdown(
        f"""
- **Population** = {POP_SIZE}  
- **Chromosome Length** = {CHROM_LEN}  
- **Generations** = {GENERATIONS}  
- **Fitness peaks at ones** = {TARGET_ONES}  
- **Max fitness** = {int(MAX_FITNESS)}  
        """
    )

    st.divider()
    st.header("Optional Controls")
    seed = st.number_input("Random seed", min_value=0, max_value=2**32 - 1, value=42)
    live = st.checkbox("Live chart while running", value=True)

    st.caption(
        f"Note: Crossover={CROSSOVER_RATE}, Mutation={MUTATION_RATE:.4f}, "
        f"Tournament k={TOURNAMENT_K}, Elitism={ELITISM}"
    )


if "_final_pop" not in st.session_state:
    st.session_state["_final_pop"] = None
    st.session_state["_final_fit"] = None


def _store_final(pop, fit):
    st.session_state["_final_pop"] = pop
    st.session_state["_final_fit"] = fit


left, right = st.columns([1, 1])

with left:
    if st.button("Run GA", type="primary"):
        problem = make_peak_40(dim=CHROM_LEN, target_ones=TARGET_ONES, max_fitness=MAX_FITNESS)

        result = run_ga(
            problem=problem,
            pop_size=POP_SIZE,
            generations=GENERATIONS,
            crossover_rate=CROSSOVER_RATE,
            mutation_rate=MUTATION_RATE,
            tournament_k=TOURNAMENT_K,
            elitism=ELITISM,
            seed=int(seed),
            stream_live=bool(live),
        )

        _store_final(result["final_population"], result["final_fitness"])

        st.subheader("Fitness Over Generations")
        st.line_chart(result["history"])

        st.subheader("Best Bit Pattern")
        st.write(f"Best fitness: **{result['best_fitness']:.2f}** (max possible = {int(MAX_FITNESS)})")

        best_bits = result["best"].astype(int)
        bitstring = "".join(map(str, best_bits.tolist()))
        st.code(bitstring, language="text")

        ones = int(np.sum(best_bits))
        st.write(f"Number of ones in best solution: **{ones} / {CHROM_LEN}**")
        st.write(f"Target ones (peak fitness): **{TARGET_ONES}**")

        # Extra summary
        final_pop = result["final_population"]
        ones_counts = final_pop.sum(axis=1)
        hits_opt = int(np.sum(ones_counts == TARGET_ONES))
        st.info(f"Individuals with exact optimum (ones = {TARGET_ONES}) in final population: **{hits_opt} / {POP_SIZE}**")

with right:
    st.subheader("Population Snapshot (final)")
    st.caption("Shows first 20 individuals with fitness and ones-count (after you run GA).")

    if st.button("Show final population table"):
        pop = st.session_state.get("_final_pop")
        fit = st.session_state.get("_final_fit")

        if pop is None or fit is None:
            st.info("Run GA first to view the final population.")
        else:
            nshow = min(20, pop.shape[0])
            df = pd.DataFrame(pop[:nshow]).astype(int)
            df["ones_count"] = pop[:nshow].sum(axis=1).astype(int)
            df["fitness"] = fit[:nshow]
            st.dataframe(df, use_container_width=True)
