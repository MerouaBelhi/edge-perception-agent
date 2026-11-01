**val split: 279 images, 63 labelled workers with a hard hat, 68 bare heads, 172 non-work headwear**

| Rules | Bare heads caught ↑ | Bare heads called 'helmet' ↓ | False alarms on hard hats ↓ | Hard hats recognised ↑ | Non-work headwear caught ↑ | Non-work headwear called 'helmet' ↓ |
|---|---:|---:|---:|---:|---:|---:|
| current rules | 68% | 0% | 6% | 81% | 75% | 0% |
| no inference from absence | 59% | 0% | 5% | 81% | 73% | 0% |
| centred helmet (dx <= 0.35 w) | 65% | 0% | 6% | 81% | 74% | 0% |
| centred helmet (dx <= 0.25 w) | 60% | 0% | 5% | 78% | 67% | 0% |
| no side margin | 68% | 0% | 6% | 81% | 75% | 0% |
| shorter head region (25 %) | 66% | 0% | 13% | 75% | 75% | 0% |
| centred 0.35 + shorter 25 % | 63% | 0% | 13% | 75% | 74% | 0% |
| helmet >= 0.20 person width | 68% | 0% | 10% | 78% | 75% | 0% |
| helmet >= 0.25 person width | 68% | 0% | 11% | 76% | 75% | 0% |

Current rules, full breakdown (rows: label, columns: system decision; 'other' = not a hard hat):

| Label | helmet | no_helmet | other | unknown | person not detected |
|---|---:|---:|---:|---:|---:|
| helmet | 51 | 2 | 2 | 4 | 4 |
| no_helmet | 0 | 44 | 2 | 5 | 17 |
| other_headwear | 0 | 4 | 125 | 5 | 38 |
| unlabelled | 0 | 21 | 13 | 30 | 94 |

**Vests** (val split: 39 labelled vests, 103 without)

| Rules | Missing vests caught ↑ | Missing vests called 'vest' ↓ | False alarms on vests ↓ | Vests recognised ↑ |
|---|---:|---:|---:|---:|
| vest rules (default) | 69% | 1% | 10% | 85% |
| fixed torso band 15-65 % (v1) | 69% | 1% | 15% | 79% |
| fixed torso band 15-95 % | 69% | 1% | 10% | 85% |
| vest >= 0.2 person width | 69% | 1% | 10% | 85% |
| vest >= 0.4 person width | 68% | 1% | 10% | 85% |

Vest, default rules (rows: label, columns: system decision):

| Label | vest | no_vest | unknown | person not detected |
|---|---:|---:|---:|---:|
| vest | 33 | 4 | 2 | 0 |
| no_vest | 1 | 71 | 6 | 25 |
| unlabelled | 0 | 162 | 29 | 128 |

**Masks** (val split: 11 labelled masks, 63 without; masks are never inferred, only detected)

| Missing masks caught ↑ | Missing masks called 'mask' ↓ | False alarms on masks ↓ | Masks recognised ↑ |
|---:|---:|---:|---:|
| 60% | 0% | 0% | 45% |

| Label | mask | no_mask | unknown | person not detected |
|---|---:|---:|---:|---:|
| mask | 5 | 0 | 5 | 1 |
| no_mask | 0 | 38 | 16 | 9 |
| unlabelled | 0 | 1 | 243 | 143 |
