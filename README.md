# Phd-Exercise-Gabriel
Evaluation of LLM Performance for Crystal Structure Prediction

------------------------------------------------------------

# REQUIREMENTS

Install all dependencies with:
python3 -m pip install pandas numpy scipy pymatgen spglib

------------------------------------------------------------

# HOW TO RUN

python3 run.py

The script will:

- Load the dataset llm_crystal_benchmark_dataset.csv
- Execute the evaluation pipeline (run.py)

Note:

Some debug messages may appear during execution. These can be ignored.
The relevant outputs are printed at the end of the execution:

- Average Similarity Score
- Average Processing Time
- Final Classification (model ranking)

Only these final results should be considered when evaluating model performance.

At the end, you can find the downloaded CSV file containing the individual similarity scores named as results.csv. These results are also in the repositorium.

# DATASET NOTES

I removed the space group index from the LLM predictions because it is redundant, as each number uniquely identifies a single space group.
------------------------------------------------------------

# METHODOLOGY

The evaluation framework consists of two main components:

1. SIMILARITY SCORE (accuracy)
2. TIME SCORE (efficiency)

------------------------------------------------------------

# 1. SIMILARITY SCORE

Each prediction is evaluated sequentially using four steps.
If a step fails, the evaluation stops at that stage.


## Step 1 (0.25/1) — Valid Response

Condition:
success == True

The model must return a valid output. A model that frequently fails is not reliable.

------------------------------------------------------------

## Step 2 (0.50/1) — Chemical Formula Match

Condition:
llm_formula == mp_formula

Matching the formula ensures the model correctly identifies the material composition, otherwise we don't need to further analyse.

------------------------------------------------------------

## Step 3 (0.75/1) — Space Group Match

Condition:
llm_space_group == mp_space_group

Even with the same composition, different space groups correspond to different structures and properties. To get the material right, it is mandotory to get the space group right.

------------------------------------------------------------

## Step 4 (0.75/1 → 1/1) — Structural Similarity

At this stage, the predicted and reference structures are already very similar. However, small geometric
differences may still exist. A good example is graphite. The interlayer distance between graphite layers 
can vary without changing the fundamental structure. Therefore, if a model  predicts the correct structure
but slightly incorrect lattice parameters, it should receive only a small penalty. This is very different from 
predicting the wrong space group.

To quantify these differences, we compute the Root Mean Square (RMS)  displacement, which measures the
average distance between corresponding atoms after optimal alignment of the two structures.

RMS = sqrt( (1/N) * sum(d_i^2) )

Where:

- d_i is the distance between corresponding atoms
- N is the number of atoms

The RMS value is calculated using pymatgen matcher.get_rms_dist(mp_struct, llm_struct) and the final score is defined as:

Score = 0.75 + 0.25 * exp(-k*RMS)

Interpretation:

- RMS ≈ 0 → perfect structural match → score ≈ 1.0
- Larger RMS or mismatch → increasing deviation → score approaches 0.75

The parameter k controls how strongly structural deviations are penalized in the scoring function.

In the exponential term exp(-k * RMS), k acts as a decay constant: larger values of k lead to a faster decrease in the score for small deviations, while smaller values result in a more tolerant scoring behavior.

The optimal choice of k depends on how strict the evaluation should be and can be adjusted depending on the application. This parameter can be further discussed and tuned during presentation. I choose k=1.5

------------------------------------------------------------

## RESULTS — SIMILARITY SCORE

{'claude-sonnet-4-6': 0.77,
 'gemini-2.5-pro': 0.65,
 'gemini-2.5-flash': 0.64,
 'gpt-5.4': 0.60,
 'gpt-5.4-mini': 0.38}

### Analysis:

- claude-sonnet-4-6 performs best and frequently reaches high structural accuracy.
- gemini-2.5-pro, gemini-2.5-flash, and gpt-5.4 show intermediate performance.
- gpt-5.4-mini often fails early and performs worst overall.

------------------------------------------------------------

# 2. TIME SCORE

## Average Processing Times:

{'gemini-2.5-pro': 37.05,
 'gemini-2.5-flash': 18.4,
 'claude-sonnet-4-6': 14.2,
 'gpt-5.4': 5.81,
 'gpt-5.4-mini': 2.99}

## Insight:

Some models have similar accuracy but very different runtimes.
For example, gpt-5.4 achieves similar accuracy to Gemini models but runs significantly faster.

------------------------------------------------------------

# FINAL SCORE

To balance accuracy and efficiency, the final evaluation metric combines the similarity score with a time-based penalty:

Final Score = 0.80 * AVERAGE_SIMILARITY_SCORE + 0.20 * exp(-beta *  AVERAGE_TIME)

The underlying idea is similar to the structural similarity scoring: penalize undesirable behavior — in this case, long computation times — using an exponential decay.

The parameter beta controls how strongly runtime is penalized:

- Large beta → stronger penalization of slow models
- Small beta → more tolerant to longer runtimes

In this implementation, beta is defined as:

beta = 1 / (average time across all models)

This normalization ensures that the time penalty is scaled relative to the overall dataset, making the final score more aligned with intuitive expectations. Models with significantly higher runtimes than average are penalized more strongly, while fast models are rewarded.

This approach allows a balanced comparison between accuracy and efficiency, while keeping the metric interpretable and adaptable.
------------------------------------------------------------

## FINAL RANKING

{'claude-sonnet-4-6': 0.70,
 'gpt-5.4': 0.62,
 'gemini-2.5-flash': 0.57,
 'gemini-2.5-pro': 0.54,
 'gpt-5.4-mini': 0.47}

------------------------------------------------------------

## KEY OBSERVATIONS

- claude-sonnet-4-6 is the best overall model.
- gpt-5.4 ranks second due to strong efficiency.
- Gemini models are accurate but slower.
- gpt-5.4-mini is fast but significantly less accurate.

------------------------------------------------------------

## CHALLENGES AND LIMITATIONS

- The main challenge in this work was comparing crystal structures represented in different forms using only a reduced basis of atoms.
If both the LLM and Materials Project (MP) structures were provided in primitive cells, this would not be an issue. In that case, the full structure could be directly constructed using:

    Structure(lattice, elements, coordinates)

However, in many cases the LLM provides structures in a conventional cell, where the reduced list of atomic positions is not sufficient to reconstruct the full structure directly from
  Structure(lattice, elements, coordinates).
  
A good example is silicon: the primitive cell contains only 2 atoms, while the conventional cubic cell requires 8 atomic positions. Using the reduced basis directly in the conventional lattice leads to an incomplete structure. This does not mean the LLM prediction is incorrect. In practice, the combination of reduced atomic positions and the space group is sufficient to define the full structure. To handle this, we use:

  Structure.from_spacegroup(lattice, elements, Wyckoff_positions)

  This method reconstructs the full structure using symmetry operations using Wyckoff_positions. However, it assumes that the input corresponds to a conventional representation and may fail or produce incorrect results if applied to primitive cells.  To decide which approach to use, we first construct the structure directly using Structure(...). We then compare the detected space group from struct.get_primitive_structure() with the reported one in the dataset:
  
  - If they match → the structure is considered consistent and it happens to be a primitive cell.
  - If they do not match → the structure is reconstructed using symmetry  

  The Structure.from_spacegroup(...) method expects atomic positions corresponding to Wyckoff positions (i.e., the minimal set of symmetry-independent atoms). However, the dataset does not always provide a true Wyckoff basis. In some cases, the list of atoms contains extra positions, which leads to duplicated atoms after applying symmetry operations.

  This issue is addressed by applying a merging step to remove duplicate or nearly overlapping atomic sites.

- In 23 out of 255 cases, this procedure fails. This occurs when the lattice parameters and atomic positions are inconsistent with the reported space group. In such cases, the symmetry expansion produces incorrect structures, and the merging step cannot recover a valid configuration leading to different number of atoms in the strucutures. As a result, structure matching fails and the RMS is infinity, leading to a Similarity Score of 0.75/1. In 30/255 cases corresponds to the cases where the final scores is 0.75, and 23/30 is because of these situation.

Given more time, I should explore more eficient ways to approch this problem. I’m worried that I may have overthought this.



------------------------------------------------------------
