# End to end: a benchmark campaign with Matchmaker deciding

ECCentric's sweep replayed on its own measurements (`experiments/end_to_end.py`): 2,971 circuits on which BP-OSD, BP-LSD, BeliefFind, Relay-BP and Tesseract were all run, 2,281 below threshold and 690 above. Each circuit is predicted by a model that never saw its code. Matchmaker skips a circuit when it predicts P(above threshold) > 0.5; a policy is charged the CPU time its decodes took in the sweep. Good enough: the chosen decoder's rate is not significantly worse than the best measured one, over the 2,039 circuits below threshold with a decode that reached its error target.

Not decoded: skipped on the prediction, stopped by a pilot, or given up; "useful" circuits have a best logical error rate below 0.3 (those between it and 0.5 are borderline).

| policy | CPU-hours | of the sweep | below threshold: good enough | not decoded | useful ones not decoded | above threshold: not decoded | CPU-hours spent above |
|---|---|---|---|---|---|---|---|
| everything | 9,917 | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 2,700 |
| always belief_find | 1,374 | 13.9% | 21.3% | 0.0% | 0.0% | 0.0% | 330 |
| always bplsd | 1,522 | 15.3% | 25.4% | 0.0% | 0.0% | 0.0% | 342 |
| always bposd | 1,863 | 18.8% | 39.5% | 0.0% | 0.0% | 0.0% | 411 |
| always relay_bp | 851 | 8.6% | 71.3% | 0.0% | 0.0% | 0.0% | 83 |
| always tesseract | 3,290 | 33.2% | 92.2% | 0.0% | 0.0% | 0.0% | 1,407 |
| matchmaker, accurate | 1,852 | 18.7% | 91.2% | 1.2% | 0.2% | 95.1% | 35 |
| matchmaker, cheapest | 1,479 | 14.9% | 91.9% | 1.2% | 0.2% | 95.1% | 26 |
| matchmaker + pilot, accurate | 1,196 | 12.1% | 90.2% | 5.3% | 0.9% | 98.6% | 11 |
| matchmaker + pilot, cheapest | 1,084 | 10.9% | 87.6% | 5.8% | 1.5% | 98.8% | 11 |
| oracle | 758 | 7.6% | 100.0% | 0.0% | 0.0% | 100.0% | 0 |
