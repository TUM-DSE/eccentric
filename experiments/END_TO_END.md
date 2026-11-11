# End to end: a benchmark campaign with Matchmaker deciding

ECCentric's sweep replayed on its own measurements (`experiments/end_to_end.py`): 2,971 circuits on which BP-OSD, BP-LSD, BeliefFind, Relay-BP and Tesseract were all run, 2,281 below threshold and 690 above. Each circuit is predicted by a model that never saw its code. Matchmaker skips a circuit when it predicts P(above threshold) > 0.5; a policy is charged the CPU time its decodes took in the sweep. Good enough: the chosen decoder's rate is not significantly worse than the best measured one, over the 2,131 circuits below threshold with a decode that reached its error target.

Not decoded: skipped on the prediction, stopped by a pilot, or given up; "useful" circuits have a best logical error rate below 0.3 (those between it and 0.5 are borderline).

| policy | CPU-hours | of the sweep | below threshold: good enough | not decoded | useful ones not decoded | above threshold: not decoded | CPU-hours spent above |
|---|---|---|---|---|---|---|---|
| everything | 10,122 | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 2,764 |
| always belief_find | 1,374 | 13.6% | 22.1% | 0.0% | 0.0% | 0.0% | 330 |
| always bplsd | 1,522 | 15.0% | 26.6% | 0.0% | 0.0% | 0.0% | 342 |
| always bposd | 1,863 | 18.4% | 39.9% | 0.0% | 0.0% | 0.0% | 411 |
| always relay_bp | 851 | 8.4% | 72.1% | 0.0% | 0.0% | 0.0% | 83 |
| always tesseract | 3,290 | 32.5% | 92.4% | 0.0% | 0.0% | 0.0% | 1,407 |
| matchmaker, accurate | 1,778 | 17.6% | 92.2% | 1.4% | 0.2% | 95.5% | 27 |
| matchmaker, cheapest | 1,385 | 13.7% | 90.8% | 1.4% | 0.2% | 95.5% | 18 |
| matchmaker + pilot, accurate | 1,414 | 14.0% | 92.7% | 3.3% | 0.3% | 98.8% | 19 |
| matchmaker + pilot, cheapest | 1,242 | 12.3% | 91.5% | 2.9% | 0.3% | 99.1% | 16 |
| oracle | 666 | 6.6% | 100.0% | 0.0% | 0.0% | 100.0% | 0 |
