# Inter-RAT handover predictor (5G / 4G)

Graduation project. The model looks at a phone's radio measurements second by second and predicts
whether it is about to change serving cell (a handover) in the next 3 seconds, then filters the
alarms so most of the ones that are left are real.

## Run it

    py -3.12 main.py

This trains the model for both regimes (5G and LTE), scores the held-out test drives, and writes the
scorecard tables and the saved models into `output/`. That's the only command you need.

## How the model works (calibrated cascade)

The prediction happens in three stages. The first stage is the actual predictor; the other two just
clean up its alarms.

1. **Base predictor** - four RandomForest experts, one per tier direction (5G->4G, 4G->5G,
   4G->legacy, legacy->4G). Every second the matching expert is picked and gives a probability of a
   handover in the next 3 s, from a 5-second window of radio and context features.
2. **Calibration** - per-arm isotonic regression rescales that probability so it means what it says
   (0.6 really is about 60%). It is fit on the training drives only, and it is the single biggest
   cut in false alarms.
3. **Episode gate** - consecutive above-threshold seconds are grouped into one alarm episode, and a
   small gradient-boosted model keeps or drops each episode from its shape (peak probability, length,
   margin over the threshold, falling RSRP). The gate threshold is set on the training drives only.

A handover here means the serving cell id actually changes (RAWCELLID on 5G, CellID on LTE) and it
is not a ping-pong (an A->B->A bounce where B is held for 3 s or less).

### Avoiding leakage
Train and test never share a drive (the split is by file, 70/30). Probabilities on the training
drives are out-of-fold, and every threshold plus the gate are chosen on training data only.

## Results (held-out test)

| | recall (caught) | precision | false alarms |
|---|---|---|---|
| 5G  | 75 % | 66 % | 0.16 / min |
| LTE | 82 % | 74 % | 0.08 / min |

Base predictor quality is ROC-AUC ~0.79 (5G) and ~0.86 (LTE). These are the operating points chosen
on the training data; `TARGET_RECALL` in `src/model.py` is the knob for trading recall against fewer
false alarms. A 20-split stress test showed the predictor itself is stable, with the exact
operational numbers moving by a few points (widest on the small 5G test set).

## KPI replay

    py -3.12 src/report.py [5g|lte|both]

This compares the recorded serving-cell sequence against an early-switch controller and reports the
drive-level KPIs (handovers/min, ping-pong share, RLF, data rate), recorded vs model. Everything is
measured from the data - there is no digital twin and no synthetic traffic. RLF here means the
downlink throughput collapsing to zero while the phone is in data state (measured, not RSRP-based).

## Layout

    main.py             trains both regimes and prints/saves the scorecards
    requirements.txt    dependencies
    input/              drive-test CSVs   (5g/ , lte/{bus,car,train}/)  -- not in the repo, see below
    src/
      data.py     loads input/, builds the 5 s feature windows + handover label, ping-pong helpers
      model.py    the model: four RF experts -> isotonic calibration -> GBM episode gate
      report.py   the scorecard and the KPI replay
    output/
      models/   model_{5g,lte}.pkl (experts + calibrators + gate)  + .json operating point
      tables/   scorecard_ml_metrics.csv, scorecard_operational.csv, scorecard_kpis.csv
      maps/     kpi_report_{5g,lte}.csv   (recorded vs early-switch KPI replay)

## Data

The datasets are not included. Download them and drop the CSVs into `input/`:
- 5G  -> `input/5g/`                    (https://github.com/uccmisl/5Gdataset)
- LTE -> `input/lte/{bus,car,train}/`   (UCC MISL 4G LTE dataset)

Built by Majd Hamdan, supervised by Prof. Ibraheem Shayea.
