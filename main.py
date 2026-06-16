# Inter-RAT handover predictor (5G / 4G), turnkey.
# One command trains the calibrated cascade for both regimes, scores the held-out test drives, and
# writes the scorecard tables + saved models into output/.
# Run:  py -3.12 main.py
import os, sys

# let "import model" / "import report" pick up the modules in src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import model
import report

REGIMES = ["5g", "lte"]

results = {}
for regime in REGIMES:
    print("\n" + "=" * 64)
    print(f"training calibrated cascade: {regime.upper()}")
    print("=" * 64)
    res = model.fit_regime(regime, save=True)
    op = model.evaluate(res["by_trace"], res["te_eps"], res["keep"])
    print(f"[{regime}] gate={res['gate']:.2f}  test recall {op['recall']:.0%}  "
          f"precision {op['precision']:.0%}  {op['fa_per_min']:.2f} false/min  "
          f"(caught {op['caught']}/{op['ho']})  -> output/models/model_{regime}.pkl")
    results[regime] = res

report.build(results)

print("\ndone. artifacts:")
print("  output/models/   model_{5g,lte}.pkl  (+ .json operating point)")
print("  output/tables/   scorecard_operational.csv, scorecard_kpis.csv")
