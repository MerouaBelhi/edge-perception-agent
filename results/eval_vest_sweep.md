**test split: 257 images, 101 labelled workers with a hard hat, 40 bare heads, 191 non-work headwear**

| Rules | Bare heads caught ↑ | Bare heads called 'helmet' ↓ | False alarms on hard hats ↓ | Hard hats recognised ↑ | Non-work headwear caught ↑ | Non-work headwear called 'helmet' ↓ |
|---|---:|---:|---:|---:|---:|---:|
| current rules | 65% | 0% | 2% | 87% | 71% | 2% |
| fixed head region on close-ups (v1) | 65% | 0% | 2% | 87% | 70% | 2% |
| no duplicate-box removal | 65% | 0% | 2% | 87% | 73% | 2% |
| no inference from absence | 60% | 0% | 2% | 87% | 68% | 2% |
| centred helmet (dx <= 0.35 w) | 62% | 0% | 2% | 87% | 70% | 2% |
| centred helmet (dx <= 0.25 w) | 55% | 2% | 2% | 77% | 65% | 2% |
| no side margin | 65% | 0% | 2% | 87% | 71% | 2% |
| shorter head region (25 %) | 65% | 0% | 2% | 86% | 71% | 2% |
| centred 0.35 + shorter 25 % | 62% | 0% | 2% | 86% | 70% | 2% |
| helmet >= 0.20 person width | 65% | 0% | 2% | 87% | 71% | 1% |
| helmet >= 0.25 person width | 65% | 0% | 4% | 85% | 71% | 1% |

Current rules, full breakdown (rows: label, columns: system decision; 'other' = not a hard hat):

| Label | helmet | no_helmet | other | unknown | person not detected |
|---|---:|---:|---:|---:|---:|
| helmet | 88 | 2 | 0 | 4 | 7 |
| no_helmet | 0 | 26 | 0 | 4 | 10 |
| other_headwear | 3 | 5 | 130 | 4 | 49 |
| unlabelled | 6 | 30 | 14 | 23 | 84 |

**Vests** (test split: 61 labelled vests, 87 without)

| Rules | Missing vests caught ↑ | Missing vests called 'vest' ↓ | False alarms on vests ↓ | Vests recognised ↑ |
|---|---:|---:|---:|---:|
| vest rules (default) | 79% | 0% | 2% | 87% |
| fixed torso band 15-65 % (v1) | 77% | 0% | 2% | 85% |
| fixed torso band 15-95 % | 79% | 0% | 0% | 85% |
| vest >= 0.2 person width | 79% | 0% | 2% | 87% |
| vest >= 0.4 person width | 78% | 0% | 5% | 84% |
| vest score >= 0.4 | 79% | 0% | 2% | 87% |
| vest score >= 0.5 | 79% | 0% | 2% | 84% |
| vest score >= 0.6 | 79% | 0% | 2% | 67% |
| vest score >= 0.7 | 79% | 0% | 2% | 56% |

Vest, default rules (rows: label, columns: system decision):

| Label | vest | no_vest | unknown | person not detected |
|---|---:|---:|---:|---:|
| vest | 53 | 1 | 3 | 4 |
| no_vest | 0 | 69 | 6 | 12 |
| unlabelled | 0 | 166 | 41 | 134 |

**Masks** (test split: 26 labelled masks, 69 without; masks are never inferred, only detected)

| Missing masks caught ↑ | Missing masks called 'mask' ↓ | False alarms on masks ↓ | Masks recognised ↑ |
|---:|---:|---:|---:|
| 77% | 0% | 0% | 73% |

| Label | mask | no_mask | unknown | person not detected |
|---|---:|---:|---:|---:|
| mask | 19 | 0 | 4 | 3 |
| no_mask | 0 | 53 | 14 | 2 |
| unlabelled | 1 | 6 | 242 | 145 |
