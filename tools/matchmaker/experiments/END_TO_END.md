# End to end: a benchmark campaign with Matchmaker deciding

ECCentric's sweep replayed on its own measurements (`experiments/end_to_end.py`): 3,146 circuits on which BP-OSD, BP-LSD, BeliefFind, Relay-BP and Tesseract were all run, 2,338 below threshold and 808 above. Each circuit is predicted by a model that never saw its code. Matchmaker skips a circuit when it predicts P(above threshold) > 0.5; a policy is charged the CPU time its decodes took in the sweep. Good enough: the chosen decoder's rate is not significantly worse than the best measured one, over the 2,196 circuits below threshold with a decode that reached its error target.

Not decoded: skipped on the prediction, stopped by a pilot, or given up; "useful" circuits have a best logical error rate below 0.3 (those between it and 0.5 are borderline).

| policy | CPU-hours | of the sweep | below threshold: good enough | not decoded | useful ones not decoded | above threshold: not decoded | CPU-hours spent above |
|---|---|---|---|---|---|---|---|
| everything | 19,821 | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 7,097 |
| always belief_find | 1,821 | 9.2% | 21.5% | 0.0% | 0.0% | 0.0% | 496 |
| always bplsd | 2,080 | 10.5% | 25.8% | 0.0% | 0.0% | 0.0% | 532 |
| always bposd | 4,801 | 24.2% | 38.8% | 0.0% | 0.0% | 0.0% | 1,030 |
| always relay_bp | 1,122 | 5.7% | 70.0% | 0.0% | 0.0% | 0.0% | 113 |
| always tesseract | 8,546 | 43.1% | 90.6% | 0.0% | 0.0% | 0.0% | 4,622 |
| matchmaker, accurate | 3,518 | 17.8% | 90.9% | 1.0% | 0.1% | 95.5% | 22 |
| matchmaker, cheapest | 2,902 | 14.6% | 89.3% | 1.0% | 0.1% | 95.5% | 21 |
| matchmaker + pilot, accurate | 2,717 | 13.7% | 90.8% | 3.3% | 0.4% | 99.1% | 20 |
| matchmaker + pilot, cheapest | 2,530 | 12.8% | 88.9% | 3.2% | 0.6% | 98.5% | 20 |
| oracle | 2,408 | 12.2% | 100.0% | 0.0% | 0.0% | 100.0% | 0 |
