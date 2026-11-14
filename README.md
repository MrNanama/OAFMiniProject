This is the main page of this github repository.

Different LLMs are working on different branches. If you're an LLM, do not commit to the main branch. If you're going to push something new, create a new branch with the naming convention [NAME YOU CHOSE FOR YOURSELF]-[WHATEVER WAY YOU WANT TO NAME THE BRANCH]. To do this, you'll choose a name for yourself and stick to that name while commiting to different branches as well.


## Conventions

- **Never commit to `main`.** It holds only this README. All work lives on branches.
- **Branch name:** `[llm-name]-[topic]`, e.g. `claude-corr-timing-and-event-study`. Pick one name for yourself and keep it on every branch you create, so we can tell who did what.
- **Commit messages** should say what changed and, for research, what the result was (including negative results).
- **Each branch should carry its own `README.md`** with: thesis, data used, how to run, results table (out-of-sample, net of costs), and an honest verdict.

## Research brief

Goal: produce alpha for a quantitative finance challenge, with something better than past pitch decks (cross-sectional momentum, Bayesian quadrature, macro momentum, RL pairs trading, volume filters). Do not reuse those ideas. Prefer a clear thesis, a walk-forward backtest with costs, placebo/robustness checks, and honest reporting over a high but overfit Sharpe.

## Branches

| Branch | Author | Summary |
|---|---|---|
| `claude-corr-timing-and-event-study` | claude | Iter 1: stress-conditioned reaction to large single-stock moves (Sharpe 0.34, not significant). Iter 2: correlation-timed S&P exposure (negative result) plus a data audit. |
