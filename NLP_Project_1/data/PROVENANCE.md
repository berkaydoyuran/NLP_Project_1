# BBC data provenance — verified 2026-10-25

Source: [Derek Greene's BBC dataset page](https://derekgreene.com/bbc/).
Author-maintained mirror description: [derekgreene/bbc-datasets](https://github.com/derekgreene/bbc-datasets).
Original raw-text download: https://derekgreene.com/datasets/bbc-fulltext.zip
Legacy project URL: http://mlg.ucd.ie/datasets/bbc.html

Provider describes 2,225 BBC articles from 2004–2005 and permits non-commercial/research use; copyright remains with the BBC. This is **not** an unrestricted public-domain/MIT dataset. The unmodified archive, including its README.TXT copyright statement, is bundled for the requested educational use. Do not relicense the article content as project code.

Archive SHA-256:
`a351031a62325b3d7ff13146eeb035c22feca8290913b5183d1ff3be35109495`

Actual format inspected: ZIP members `bbc/<category>/<number>.txt`, one UTF-8 text file per article. All article files were decoded with UTF-8 successfully. The title is in the text; there are **no CSV field names**. Internal loader fields are our own `id`, `label`, `text`. Label comes from the parent directory and is kept out of model messages.

| Label | Verified articles |
|---|---:|
| business | 510 |
| entertainment | 386 |
| politics | 417 |
| sport | 511 |
| tech | 401 |
| Total | 2225 |

The raw fulltext archive is used, not the stemmed term-frequency matrix. Only CRLF normalization and outer whitespace removal are applied. Sampling: sorted identifiers, Python `random.Random(42)`, 20 without replacement per category, round-robin ordering. This is a balanced stratified evaluation sample, not the natural corpus distribution.

Provider's requested citation: D. Greene and P. Cunningham (2006), “Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering,” Proceedings of ICML 2006. [Author's paper](https://derekgreene.com/papers/greene06icml.pdf). That paper is provenance for the dataset, not evidence that this LLM approach achieves any particular score.
