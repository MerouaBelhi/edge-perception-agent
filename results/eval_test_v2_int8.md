**test split: 257 images, 101 labelled workers with a hard hat, 40 bare heads, 191 non-work headwear**

| Rules | Bare heads caught ↑ | Bare heads called 'helmet' ↓ | False alarms on hard hats ↓ | Hard hats recognised ↑ | Non-work headwear caught ↑ | Non-work headwear called 'helmet' ↓ |
|---|---:|---:|---:|---:|---:|---:|
| current rules | 65% | 0% | 3% | 81% | 74% | 1% |

Current rules, full breakdown (rows: label, columns: system decision; 'other' = not a hard hat):

| Label | helmet | no_helmet | other | unknown | person not detected |
|---|---:|---:|---:|---:|---:|
| helmet | 82 | 3 | 0 | 4 | 12 |
| no_helmet | 0 | 26 | 0 | 5 | 9 |
| other_headwear | 1 | 7 | 134 | 4 | 45 |
| unlabelled | 3 | 34 | 9 | 19 | 92 |

**Vests** (test split: 61 labelled vests, 87 without)

| Rules | Missing vests caught ↑ | Missing vests called 'vest' ↓ | False alarms on vests ↓ | Vests recognised ↑ |
|---|---:|---:|---:|---:|
| vest rules (default) | 74% | 0% | 7% | 80% |

Vest, default rules (rows: label, columns: system decision):

| Label | vest | no_vest | unknown | person not detected |
|---|---:|---:|---:|---:|
| vest | 49 | 4 | 2 | 6 |
| no_vest | 0 | 64 | 10 | 13 |
| unlabelled | 1 | 165 | 36 | 139 |

**Masks** (test split: 26 labelled masks, 69 without; masks are never inferred, only detected)

| Missing masks caught ↑ | Missing masks called 'mask' ↓ | False alarms on masks ↓ | Masks recognised ↑ |
|---:|---:|---:|---:|
| 71% | 0% | 0% | 69% |

| Label | mask | no_mask | unknown | person not detected |
|---|---:|---:|---:|---:|
| mask | 18 | 0 | 4 | 4 |
| no_mask | 0 | 49 | 17 | 3 |
| unlabelled | 0 | 6 | 237 | 151 |
