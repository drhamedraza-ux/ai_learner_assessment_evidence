# Paper 2 — 12 standalone notebooks (4 RQs × 3 datasets)

One notebook per research question per dataset. Each is fully self-contained — no shared file, no
`DATASET` switch to remember. Open it, check the paths in the setup cell, run top to bottom.

```
P2_RQ1_assistments_2021.ipynb   P2_RQ1_ednet_flat.ipynb   P2_RQ1_junyi.ipynb   -> Table 1 + subset
P2_RQ2_assistments_2021.ipynb   P2_RQ2_ednet_flat.ipynb   P2_RQ2_junyi.ipynb   -> Tables 2 & 3
P2_RQ3_assistments_2021.ipynb   P2_RQ3_ednet_flat.ipynb   P2_RQ3_junyi.ipynb   -> Tables 4 & 5
P2_RQ4_assistments_2021.ipynb   P2_RQ4_ednet_flat.ipynb   P2_RQ4_junyi.ipynb   -> Tables 6 & 7
```

## Run order (per dataset)

For each dataset, **run RQ1 first** — it writes `audited_subset_<dataset>.json`, which RQ2/RQ3/RQ4 for
that same dataset read to know which PCDT dimensions to use. Then run RQ2, RQ3, RQ4 in any order.

```
assistments_2021:  RQ1 -> RQ2 -> RQ3 -> RQ4
ednet_flat:        RQ1 -> RQ2 -> RQ3 -> RQ4
junyi:             RQ1 -> RQ2 -> RQ3 -> RQ4
```

That's 12 runs total. Each RQ2/3/4 notebook checks that the subset it loads matches its dataset and
refuses a mismatched one.

## Paths (already baked in — edit only if yours differ)

The setup cell in every notebook has:

```python
DATA_PATHS = {
  "assistments_2021": dict(plogs=".../feb_to_apr/plogs.csv", details=".../feb_to_apr/pdets.csv"),
  "ednet_flat":       dict(flat=".../EdNet/ednet_kt1_flat.csv"),
  "junyi":            dict(log=".../Junyi/Log_Problem.csv", content=".../Junyi/Info_Content.csv"),
}
CACHE_DIR = ".../PCDT_P2_cache"
```

Only the current notebook's dataset entry needs to be correct.

## IMPORTANT — delete the stale subset first

Earlier EdNet runs left a generic `audited_subset.json` in your cache dir. Before running, delete it:

```
rm -f <CACHE_DIR>/audited_subset.json
```

Otherwise an old file can shadow the correct per-dataset one. (The notebooks now guard against loading a
subset whose dataset tag doesn't match, but removing the stale file avoids the warning entirely.)

## Fixes included in these notebooks (vs the combined notebook you ran)

1. **B_t now uses the real behavioural signal per dataset.** ASSISTments and Junyi have hint + attempt
   columns, so B_t is audited as a hint+attempt+latency struggle signal; EdNet has neither, so B_t falls
   back to latency-only. Previously the audit was hardcoded latency-only, which wrongly rejected B_t on
   ASSISTments/Junyi. You can see which was used in Table 1's "Proxy set" column.
2. **BKT no longer crashes with NaN.** The mastery belief is clamped after every update (long real
   sequences were saturating it and producing NaN), the AUC/bootstrap code drops any non-finite
   prediction, and BKT never reuses cached predictions (an older cache could hold pre-fix NaNs).
3. **Per-dataset subset file + dataset guard**, and the duplicate-`problem_id` fix in the ASSISTments
   skill join. All carried over from the combined notebook.

## Two things to verify on your real data

- **Timestamp units / span.** Each loader prints a median-learner-span line. It should read in **weeks**
  (ASSISTments/Junyi) or however long your EdNet window is — not minutes. Junyi's Taiwan-timezone strings
  and EdNet's milliseconds are the two most likely to misparse; a wrong unit silently corrupts RQ3's gap
  analysis.
- **Skill coverage.** Table 1 / the loader prints how many distinct skills resulted. For ASSISTments this
  comes from the `problem_id -> skills` join; if it collapses to ~1 skill / mostly `UNK`, your log's
  `problem_id`s don't match the details file.

## Cosmetic notes (harmless)

- The RQ1 notebooks still print "EdNet-KT1" in Table 1's title and one diagnostic line regardless of
  dataset — the audit code was originally written for EdNet. The **verdicts are computed on the actual
  selected dataset**; check the provenance cell, which reports the true dataset. Trust the "Proxy set"
  column (it correctly says "struggle signal" vs "latency-only") over the title.
- RQ4 may print one stray "EdNet-SHAPED FIXTURE" banner even on real data; its data is the real dataset
  (provenance cell confirms `using_fixture=False`).


## Output folders (one per notebook)

Each notebook writes all of its artifacts into its own folder, so the 12 runs never collide:

```
<CACHE_DIR>/results/RQ1_assistments_2021/   table1.csv, table1.md, subset copy
<CACHE_DIR>/results/RQ2_assistments_2021/   table2*, table3*, *.npz caches, *.tex
<CACHE_DIR>/results/RQ3_assistments_2021/   table4*, table5*, *.npz
<CACHE_DIR>/results/RQ4_assistments_2021/   table6*, table7*, heatmap.png
...and the same for _ednet_flat and _junyi
```

- `CACHE_DIR` is set in the setup cell (default `/home/hamed/Downloads/Research/P02/PCDT_P2_cache`).
  Change it there and every folder moves with it.
- **The audited subset is written to `SUBSET_DIR` (= `CACHE_DIR`), NOT the per-notebook folder**, so the
  RQ2/RQ3/RQ4 notebooks for the same dataset can find it. A copy is also placed in RQ1's own folder for
  your convenience. This is why the RQ1 -> RQ2/3/4 handoff still works with separated output folders.
- Tables, caches (`.npz`), figures, and caption files all land in the per-notebook `OUTPUT_DIR`.
