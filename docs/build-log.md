# Build log

Everything done on this project, in order, so it can be explained on the page (web/index.html, "How this was made").

## Nov 13 2026
1. **Data.** SketchGraphs (Princeton; sketches from public Onshape documents). Files: sg_t16_train / validation / test,
   9,179,789 / 315,228 / 313,271 sketches of at most 16 shapes. No hand labels: the next shape drawn is the answer.
2. **Tokens.** Each shape = a kind token (line, circle, arc, point; separate tokens for helper/construction shapes)
   plus coordinates on a 64 x 64 grid; shapes kept in the order drawn. 75 tokens in all, at most 128 per sketch.
3. **Cleaning.** Usable (two or more shapes that survive the grid): 7,055,701 train. Distinct: 2,512,441.
   Test: 241,558 usable, 121,797 distinct, 76,694 after removing those also in training (37% of the distinct ones were).
   Validation: 243,092 usable, 123,394 distinct, 77,699 kept.
4. **Model.** GPT written out in model.py: 6 layers, width 256, 4 heads, 4,810,315 weights.
5. **Training.** One RunPod RTX 4090 ($0.89/hour, capped at 8 hours / $8). AdamW, weight decay 0.05, learning rate 1e-3
   with warm-up and cosine fall, batch 1024, 60,000 steps (about 24 passes). Best validation loss is kept (early stopping).
   Error on unseen sketches: 4.37 at the start, 1.20 at step 2,000, 1.11 at step 6,000.
6. **Scoring.** evaluate.py: hide the next shape at every point of 20,000 test sketches; exact / top-3 / kind / near, and
   precision and recall as the confidence bar rises. (Results: pending, filled in when training ends.)
7. **Browser.** web/model.js is the same model in plain JavaScript with a memory of the shapes already read.
   test/parity.mjs: 10 cases, largest score difference 0.0001, same five guesses in all 10, about 130 ms a guess.
8. **Page.** web/index.html + app.js: draw lines, circles, arcs, points; dashed blue guess; Tab takes it.

## Things that went wrong
- The pod image lacked rsync and the data reader's extras (matplotlib, lz4); installed by hand.
- prep.py first used 96 worker processes because the machine reports 128 processors, but the pod only gets about 18.
  They starved each other and nothing finished in four minutes; 16 workers finished the lot in about two minutes.
- The GitHub sign-in on this computer had expired; Clark signed in again.

## Not done yet / known limits
- Constraints (parallel, equal, mirror ...) are in the data but not modelled; only shapes are.
- Training sketches are scaled to fill the grid, so a small drawing in the middle of the page is unusual to the model.
- At most 16 shapes per sketch.
