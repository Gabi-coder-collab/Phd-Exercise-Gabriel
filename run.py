import pandas as pd
import numpy as np
import ast
import re

from pymatgen.core import Structure, Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


###########################################################
# STEP 1 — Load dataset
###########################################################

def load_data(path):
    return pd.read_csv(path, engine="python", sep=",", quotechar='"')


###########################################################
# STEP 2 — Clean space groups
###########################################################

def clean_space_group(sg):
    if isinstance(sg, str):
        return re.sub(r"\s*\(.*?\)", "", sg).strip()
    return sg


def preprocess(df):
    df["llm_space_group"] = df["llm_space_group"].apply(clean_space_group)
    df["mp_space_group"]  = df["mp_space_group"].apply(clean_space_group)

    for col in [
        "llm_lattice_params",
        "mp_lattice_params",
        "llm_atomic_positions",
        "mp_atomic_positions"
    ]:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    return df


###########################################################
# STEP 3 — Scoring
###########################################################

def compute_score(row, k=1.0):

    score = 0.0

    # Step 1: valid response
    if not row.get("success", False):
        return 0.0
    score += 0.25

    # Step 2: formula match
    if str(row.get("llm_formula")) != str(row.get("mp_formula")):
        return score
    score += 0.25

    # Step 3: space group match
    if row.get("llm_space_group") != row.get("mp_space_group"):
        return score
    score += 0.25

    try:
        # --- MP structure ---
        mp_lat = list(row["mp_lattice_params"].values())
        mp_el  = [d["element"] for d in row["mp_atomic_positions"]]
        mp_pos = [d["position"] for d in row["mp_atomic_positions"]]

        mp_struct = Structure(Lattice.from_parameters(*mp_lat), mp_el, mp_pos)

        mp_sg = row["mp_space_group"]
        mp_sg_detected = SpacegroupAnalyzer(mp_struct, symprec=1e-2).get_space_group_symbol()

        if mp_sg_detected != mp_sg:
            mp_struct = Structure.from_spacegroup(
                mp_sg,
                Lattice.from_parameters(*mp_lat),
                mp_el,
                mp_pos
            )
            mp_struct.merge_sites(tol=0.1, mode="delete")

        # --- LLM structure ---
        llm_lat = list(row["llm_lattice_params"].values())
        llm_el  = [d["element"] for d in row["llm_atomic_positions"]]
        llm_pos = [d["position"] for d in row["llm_atomic_positions"]]

        llm_struct = Structure(Lattice.from_parameters(*llm_lat), llm_el, llm_pos)

        llm_sg = row["llm_space_group"]
        llm_sg_detected = SpacegroupAnalyzer(llm_struct, symprec=1e-2).get_space_group_symbol()

        if llm_sg_detected != llm_sg:
            llm_struct = Structure.from_spacegroup(
                llm_sg,
                Lattice.from_parameters(*llm_lat),
                llm_el,
                llm_pos
            )
            llm_struct.merge_sites(tol=0.1, mode="delete")

        # --- RMS matching ---
        matcher = StructureMatcher(stol=1)
        res = matcher.get_rms_dist(mp_struct, llm_struct)

        if res:
            rms = res[0] if isinstance(res, tuple) else res
            score += 0.25 * np.exp(-k * rms)

    except Exception as e:
        print(f"Error on row {row.name}: {e}")

    return round(score, 4)


###########################################################
# STEP 4 — Evaluation
###########################################################

def evaluate(df):

    df["score"] = df.apply(lambda r: compute_score(r, k=1), axis=1)

    scores = df.groupby("llm_model")["score"].mean()
    times  = df.groupby("llm_model")["processing_time"].mean()

    k = 1 / times.mean()
    final = 0.80 * scores + 0.20 * np.exp(-k * times)

    return (
        scores.sort_values(ascending=False).round(2),
        times.sort_values(ascending=False).round(2),
        final.sort_values(ascending=False).round(2),
    )


###########################################################
# MAIN
###########################################################

def main():
    df = load_data("llm_crystal_benchmark_dataset.csv")
    df = preprocess(df)

    scores, times, final = evaluate(df)

    print("\n=== Average Similarity Score ===")
    print(scores.to_dict())

    print("\n=== Average Processing Time ===")
    print(times.to_dict())

    print("\n=== Final Ranking ===")
    print(final.to_dict())


if __name__ == "__main__":
    main()