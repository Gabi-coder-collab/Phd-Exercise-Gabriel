import pandas as pd
import re
import numpy as np
import ast
import pymatgen
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure_matcher import StructureMatcher


###########################################################################################################
############################################## DATA HANDLING ##############################################
###########################################################################################################



###Step1###

df0 = pd.read_csv(
    "llm_crystal_benchmark_dataset.csv",
    engine="python",
    sep=",",
    quotechar='"'
)


###Step2###

df1 = df0

def clean_space_group(sg):
    if isinstance(sg, str):
        return re.sub(r"\s*\(.*?\)", "", sg).strip()
    return sg

df1["llm_space_group"] = df1["llm_space_group"].apply(clean_space_group)
df1["mp_space_group"]  = df1["mp_space_group"].apply(clean_space_group)


###Step3###


df=df1

df["llm_lattice_params"] = df["llm_lattice_params"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

df["mp_lattice_params"] = df["mp_lattice_params"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

df["llm_atomic_positions"] = df["llm_atomic_positions"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

df["mp_atomic_positions"] = df["mp_atomic_positions"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)


###########################################################################################################
############################################## SIMILARITY SCORING ##########################################
###########################################################################################################



def Score(row, k=1.0):
    print()
    print(f"DEBUG Row {row.name}:")
    actual_score = 0.0
    
    ###Step1###
    if not row.get("success", False):
        print("Insuss")
        return 0.0
    actual_score = 0.25
    print("Sucess")

    ###Step2###
    if str(row.get("llm_formula")) != str(row.get("mp_formula")):
        print("Different formula")
        return actual_score
    actual_score += 0.25
    print("Same formula")

    llm_sg = row.get("llm_space_group")
    mp_sg  = row.get("mp_space_group")

    ###Step3###
    if llm_sg != mp_sg:
        print("Different space group")
        return actual_score
    actual_score += 0.25
    print("Same Space group")

    ###Step4###
    try:
        # =========================
        # --- MP ---
        # =========================
        mp_lat = list(row["mp_lattice_params"].values())
        mp_el  = [d["element"] for d in row["mp_atomic_positions"]]
        mp_pos = [d["position"] for d in row["mp_atomic_positions"]]

        mp_struct = Structure(Lattice.from_parameters(*mp_lat), mp_el, mp_pos)

        mp_sg2 = SpacegroupAnalyzer(mp_struct, symprec=1e-2).get_space_group_symbol()
        print("Raw MP SG match:", mp_sg == mp_sg2)

        if mp_sg2 != mp_sg:
            try:
                mp_struct = Structure.from_spacegroup(
                    mp_sg,
                    Lattice.from_parameters(*mp_lat),
                    mp_el,
                    mp_pos
                )
                mp_struct.merge_sites(tol=0.1, mode="delete")
            except:
                pass

        # =========================
        # --- LLM ---
        # =========================

        llm_lat = list(row["llm_lattice_params"].values())
        llm_el  = [d["element"] for d in row["llm_atomic_positions"]]
        llm_pos = [d["position"] for d in row["llm_atomic_positions"]]

        llm_struct = Structure(Lattice.from_parameters(*llm_lat), llm_el, llm_pos)

        llm_sg2 = SpacegroupAnalyzer(llm_struct, symprec=1e-2).get_space_group_symbol()
        print("RAW LLM SG match:", llm_sg == llm_sg2)

        if llm_sg2 != llm_sg:
            try:
                llm_struct = Structure.from_spacegroup(
                    llm_sg,
                    Lattice.from_parameters(*llm_lat),
                    llm_el,
                    llm_pos
                )
                llm_struct.merge_sites(tol=0.1, mode="delete")
            except:
                pass

        # =========================
        # --- PRIMITIVE CHECK ---
        # =========================
        try:
            mp_prim  = mp_struct.get_primitive_structure()
            llm_prim = llm_struct.get_primitive_structure()


            # =========================
            # --- For Debbuging  ---
            # =========================

            n_mp_prim  = len(mp_prim)
            n_llm_prim = len(llm_prim)

            if n_mp_prim != n_llm_prim:
                print(f"DEBUG Row {row.name}: Primitive mismatch (MP: {n_mp_prim} vs LLM: {n_llm_prim})")

        except Exception as e:
            print(f"DEBUG Row {row.name}: Primitive error: {e}")

        # =========================
        # --- MATCHING ---
        # =========================
        matcher = StructureMatcher(stol=1)

        res = matcher.get_rms_dist(mp_struct, llm_struct)

        if res:
            rms_val = res[0] if isinstance(res, tuple) else res
            actual_score += 0.25 * np.exp(-k * rms_val)

    except Exception as e:
        print(f"DEBUG Row {row.name}: Error: {e}")

    return round(actual_score, 4)


df['score'] = df.apply(lambda r: Score(r, k=1.5), axis=1)
df.to_csv("results.csv", index=False)





###########################################################################################################
############################################## Final Classification ########################################
###########################################################################################################

# --- Base metrics ---
scores_series = df.groupby('llm_model')['score'].mean()
times_series  = df.groupby('llm_model')['processing_time'].mean()

# --- Final score ---
k = 1 / times_series.mean()
final_score = 0.80 * scores_series + 0.20 * np.exp(-k * times_series)

# --- Sort each independently ---
scores_series = scores_series.sort_values(ascending=False)
times_series  = times_series.sort_values(ascending=False)
final_score   = final_score.sort_values(ascending=False)

# --- Round ---
scores_series = scores_series.round(2)
times_series  = times_series.round(2)
final_score   = final_score.round(2)

# --- Convert to dicts ---
scores_dict = scores_series.to_dict()
times_dict  = times_series.to_dict()
final_dict  = final_score.to_dict()


print()
print()
print()
print()
print()


print()
print()
print()
print()
print()


print()
print()
print()
print()
print()


print("Average Similarity Score")
print(scores_dict)
print()

print("Average Times")
print(times_dict)
print()

print("Final Classification")
print(final_dict)
print()