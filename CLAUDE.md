# Rules for LLMs working in this repo (read before doing any research)

## Name and branches
- Use your assigned name as the branch prefix: `[name]-[topic]`. Never commit to `main` unless the user explicitly says so.

## Rule 1: Train/test discipline (mandatory)
Never discover, tune, or choose anything on all of the available data.
1. **Before looking at any data**, fix a time split and write it down (in the branch README and in `oaf/split.py`): default is the first ~70% of the history for TRAIN and the last ~30% for TEST, with a 1-month embargo between them.
2. **Explore, scan signals, pick parameters, design rules and diagnose losses ONLY on TRAIN.** Load data through `oaf.split.train(...)`. Do not even print summary stats of the test period.
3. **Freeze** the full strategy (signal, parameters, filters, sizing, costs) and commit it BEFORE touching TEST.
4. **Touch TEST once** via `oaf.split.unlock_test("<why>")` (this logs the access). Report the result whatever it is. If it fails, do NOT go back and tweak and re-test on the same TEST; that makes it train data. Declare it burned and say so.
5. If a diagnosis of a TEST-period loss motivates a new rule, that rule is contaminated: label it as such, and it needs a fresh holdout (new period, new venue, new universe, or forward paper-trading).
6. Walk-forward (expanding/rolling refit on past data only, with purge) is fine inside TRAIN and TEST, but does not replace the held-out TEST.

## Rule 2: Honest reporting
- Count every signal / parameter / variant tried and report the count; report deflated Sharpe with that count.
- State explicitly which numbers are train, test, or contaminated. Put negative results in the README too.
- Never present in-sample numbers as expected performance.

## Rule 3: Data hygiene
- No look-ahead: features at close t use data through t only; positions earn from t+1. Purge overlapping labels.
- Prefer survivorship-free / point-in-time data; if impossible, say how it biases the result.
- Include costs and financing; report breakeven cost.
