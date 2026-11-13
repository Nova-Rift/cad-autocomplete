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
   precision and recall as the confidence bar rises. (Results: item 11.)
7. **Browser.** web/model.js is the same model in plain JavaScript with a memory of the shapes already read.
   test/parity.mjs: 10 cases, largest score difference 0.0001, same five guesses in all 10, about 130 ms a guess.
8. **Page.** web/index.html + app.js: draw lines, circles, arcs, points; dashed blue guess; Tab takes it.

## Nov 13 2026, later
9. **The page-edge flaw.** Trying the step-6,000 model in the page: its guesses ran to the page edge (a circle start got a
   huge circle; half a box got a short line to the edge). Cause: every training sketch is scaled to fill the 64 x 64 grid,
   so the model learns that the finished sketch must touch the edges. This is a training/serving mismatch: people drawing
   on the page do not fill it. First run stopped at step 8,000 (kept as run-first).
10. **Second run.** train.py now shrinks each training sketch by a random factor 0.45 to 1 and slides it to a random spot
    (80% of sketches; 20% left as stored). Same settings otherwise. 60,000 steps in 72.5 minutes. Best copy = step 60,000.
    Loss at the end: training 1.2157, unseen (moved) 1.2236, unseen (as stored) 1.0786. Both still falling.
11. **Scores** (5,000 test sketches, 31,859 hidden shapes each; evaluate.py; "moved" = test sketches shrunk and shifted):

    | model | test as stored: exact / top 3 | test moved: exact / top 3 |
    |---|---|---|
    | first run, step 8,000 (no moving) | 40.6% / 52.1% | 26.4% / 33.1% |
    | second run, step 8,000 | 38.7% / 50.0% | 30.9% / 40.4% |
    | second run, step 60,000 (final) | 42.6% / 54.4% | 34.4% / 44.5% |

    Final model, test as stored: right kind of shape 74.5%; within one grid step on every coordinate 46.1%.
    By kind (exact / top 3): line 47% / 58%, circle 24% / 38%, arc 32% / 41%, point 31% / 51%.
    Confidence bar 50%: guesses shown on 39% of steps, precision 77%, recall 30%. Bar 90%: shown 18%, precision 97%.
    The fix cost about 2 points on page-filling sketches at equal training and gained 4.5 on moved ones.
12. **Checkpoints.** Clark did not want them on his computer: ckpt_watch.py on the pod sent every improved copy to
    Cloudflare R2 (`cad-autocomplete/run/step-*.pt`, 30 copies), plus results, pictures and the token rows.
13. **Page on the site.** musclememorybench.com/sketch/: first as a bare page, then inside the site frame; content in
    web/body.html, copied by research-lab tools/cad-autocomplete/sync_site.py. System design drawn as a diagram.
14. **Guesses on the board.** All three guesses are drawn on the board with numbers; a strip above steps through them.
    When the user clicks to start a shape, the search is forced to agree with what is fixed so far (kind and start
    point; for an arc also the end), so the guesses finish that shape (model.js `suggest(tokens, width, forced)`).
15. **Cost.** Pod rented 1.85 hours at $0.89: about $1.64. Pod deleted.

16. **Page as a showcase.** Clark: the page is a project to show people, so the study-guide parts ("Lesson" labels, a
    "what a project like this takes" section, the name of the book being studied) were removed. The system design sits
    above "How this was made". Results, pictures and a "Not done yet" note were added with the finished model.

## Things that went wrong
- The pod image lacked rsync and the data reader's extras (matplotlib, lz4); installed by hand.
- prep.py first used 96 worker processes because the machine reports 128 processors, but the pod only gets about 18.
  They starved each other and nothing finished in four minutes; 16 workers finished the lot in about two minutes.
- The GitHub sign-in on this computer had expired; Clark signed in again.
- evaluate.py reused the name `out` for the output path and for a token list, so the moved test crashed when saving; renamed.
- The first training chart compared training error on moved sketches with unseen error on unmoved ones; redrawn like with like.
- The site build's size limit (60 MiB) was hit by the 9 MiB model file; the page got its own 12 MiB allowance.

## Not done yet / known limits
- Constraints (parallel, equal, mirror ...) are in the data but not modelled; only shapes are.
- Training sketches are scaled to fill the grid, so a small drawing in the middle of the page is unusual to the model.
- At most 16 shapes per sketch.
