# Quick README

---

### Experiment command

```
python run.py --dataset adult --sens_keys sex,race,age,marital-status --data_dir data/raw/ --method dr --lambda_fair 1.00 --union_mode apriori_forward  --n_low_frac 0.2
```

### data
- Adult `data/raw/adult.csv`
    - --dataset adult --sens_keys sex,race,age,marital-status --sens_thresh 0.5

### Arguments

- DRAF
    - lambda_fair: Controls fairness regularization.
    - n_low_frac: Controls minimum subgroup size (gamma in the paper).

---------

