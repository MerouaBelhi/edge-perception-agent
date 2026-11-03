**test split: 257 images, 101 labelled workers with a hard hat, 40 bare heads, 191 non-work headwear**

| Rules | Bare heads caught ↑ | Bare heads called 'helmet' ↓ | False alarms on hard hats ↓ | Hard hats recognised ↑ | Non-work headwear caught ↑ | Non-work headwear called 'helmet' ↓ |
|---|---:|---:|---:|---:|---:|---:|
| current rules | 65% | 0% | 2% | 87% | 71% | 2% |

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
