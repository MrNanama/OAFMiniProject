# Presentation: Rebalancing Flows (OAF-style pitch, 16 slides)

`OAF_Rebalancing_Flows.pptx` - pitch deck for strategy s15 (rebalancing flows), with speaker notes on every slide.
All numbers come from the frozen backtests (post-freeze reporting):

```bash
python presentation/extract_data.py          # -> presentation/deck_data.json (needs the FROZEN vault data)
NODE_PATH=<node_modules with pptxgenjs react react-dom react-icons sharp> \
PPTX_SKILL_DIR=<pptx skill dir providing scripts/apply_theme.js> node presentation/build_deck.js
```
Charts are native PowerPoint charts (editable); amber = stocks (SPY), teal = bonds (TLT) throughout.
