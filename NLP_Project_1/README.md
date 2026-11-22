# Local structured news classification and entity extraction

Berkay Doyuran · NLP Project I · November 2026

**Final status (22 November): both 100-input runs are complete. Metrics were recomputed from the original 200 responses and the final 100/100 human-reviewed references. Raw/cleaned parsing: 100%/100%; schema validity: 95%/100%; strict classification accuracy: 67%/70%; entity micro-F1: 0.5236/0.5417 (prompt/schema). The four-page `report.pdf` contains the final results and error analysis.**

The reference file contains 1,382 distinct document-level entity pairs. AI-assisted annotation suggestions were reviewed and confirmed by Berkay Doyuran. Review flags record the user's confirmations; automatic suggestions alone are not human-reviewed gold.

## Objective and methodology

Classify each news article and extract PERSON, ORG and LOC in one JSON answer. Compare the same 100 inputs under prompt-only generation (no `format` field) and schema-constrained generation (`format=SCHEMA`). All messages and generation parameters are identical between conditions. There is no task-specific model training. All experimental model inference uses local Ollama; no cloud inference API is called by the project.

## Dataset

BBC Full Text: 2,225 articles, five labels (business, entertainment, politics, sport, tech). The existing 100-item sample, 20/class, was selected with Python Random seed 42. **This version loads that fixed sample; it never resamples it.** Input is the original case-preserved UTF-8-safe prefix of at most 4,000 bytes. This may cut a word or omit evidence. Folder labels are references, never passed to the model.

See `data/PROVENANCE.md` for source, checksum, copyright and provider citation. The unmodified archive includes the original copyright statement. BBC article content remains copyrighted and is provided for non-commercial/research use; do not relicense it as program code.

## Structure

- `main.py`: input loading, a small command selector, sequential paired runs, checkpoint saving, evaluation and fresh live tests.
- `inference.py`: unchanged task prompts/schema, local Ollama calls, raw parsing, conservative cleaning and validation.
- `evaluation.py`: human-reference checks, exact entity set scoring and classification/format metrics.
- `data/`: original corpus, fixed inputs, gold annotation file, sourced live inputs and provenance.
- `results/`: two main JSON files once run, and a comparison JSON/CSV. Live takes get separate folders.
- `report.pdf`: four-page final report, including reviewed-reference NER metrics and paired error analysis.

Learning material, annotation UI, old source archive and offline verification evidence are provided separately. They are not required imports or runtime dependencies.

## Model configuration

Model tag `llama3.2:3b`; actual main-run metadata: Q4_K_M, Ollama 0.40.16 (recorded in both uploaded mode files). A fresh check records current metadata; it is not fabricated from these values. Actual run metadata includes digest, quantization, version and complete model details.

| Setting | Value |
|---|---|
| temperature / seed | 0.0 / 42 |
| num_ctx / num_predict | 8192 / 1024 |
| top_k / top_p | 40 / 0.9 |
| repeat_penalty / repeat_last_n | 1.1 / 256 |
| input limit / timeout | 4000 bytes / 300 seconds |

The updated system/user prompts and repetition settings from the successful smoke fix are retained exactly. Prompt and settings appear in `inference.py`; main-run metadata stores them too. Temperature 0 is not a guarantee of reproducibility across different runtimes or hardware.

## Installation (Windows PowerShell)

Extract into a separate folder to preserve the previous project. Open its `NLP_Project_1` folder in VS Code. Python 3.11 or 3.12 is suitable.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Ollama and the existing model must be installed and running. The code does not download a model automatically. The included BBC archive means no dataset download is necessary.

## Execution

```powershell
# Optional metadata check; no news inference
.\.venv\Scripts\python.exe main.py check

# Main experiment: 100 prompt-only + 100 schema calls, then evaluation
.\.venv\Scripts\python.exe main.py run

# Re-evaluate existing answers after exporting reviewed gold; no model calls
.\.venv\Scripts\python.exe main.py evaluate

# Video rehearsal/recording: 5 samples x 2 modes x 2 temperatures; fresh calls
.\.venv\Scripts\python.exe main.py live
```

Run only one main process at a time. Each response is saved before the next call. If interrupted, repeat `run` with unchanged conditions; recorded attempts, including failed requests, are not repeated. Do not alter code/settings while a run is active. If some calls failed, preserve the files and inspect the cause before planning a retry. The two mode files retain their own original raw responses.

The live command always uses a new folder and makes 20 fresh calls. Source texts and source URLs are recorded. Expected labels in the live sample file are provisional interpretations, not a substitute for the 100-item benchmark. An expected-error sample is not guaranteed to fail. Accuracy of live examples must be reviewed from their content, not inferred from format flags.

### Optional import of previous full/partial runs

Not needed for the student's currently confirmed state. For an existing post-fix main run:

```powershell
.\.venv\Scripts\python.exe main.py import "D:\Downloads\NLP_Project_1_Delivery\NLP_Project_1"
```

This is offline. It verifies the known original implementation, requests, inputs and checkpoint provenance before repackaging the original responses. It preserves the original manifest, skips no validation and refuses conflicting destination runs/annotation work. A different legacy version requires inspection, not an automatic rerun. Moving functions to different files alone does not invalidate matching model responses. Pilot runs are not imported as main runs.

## Human annotation

The supplied `data/gold_entities.json` contains the final 100/100 reviewed inputs. The separately supplied `annotation.html` supports highlighted PERSON/ORG/LOC suggestions, explanations, edits and explicit per-article confirmation. Browser changes are saved locally; export JSON backups because browser storage may be cleared or unavailable. Any later annotation edits require renewed review and evaluation. Copy the reviewed export to `data/gold_entities.json`, then run `evaluate`; no new model calls are needed.

Exact case-sensitive `(surface text, type)` pairs must match. Keep different surface forms distinct; no alias merging or partial matching. Deduplicate identical pairs within each document. Prefer the longest named span at an overlapping occurrence; a nested name independently mentioned elsewhere may also be included. Exclude dates, numbers, products, nationalities and generic nouns. This is document-level entity-set F1, not token/offset mention-level F1. One reviewer means no inter-annotator agreement measurement.

## Evaluation

Raw success applies default `json.loads` to the untouched answer. Strict parsing additionally rejects nonfinite values and duplicate keys. Cleaning extracts only a single complete JSON container surrounded by text/fences. It never closes truncated JSON, chooses among multiple containers, fills missing fields or repairs category/entity content. Schema validation follows parsing. The stricter cleaned rate can be lower than permissive default parsing.

Each primary format/classification rate has denominator **all 100 planned inputs**. Invalid/missing answers count as unsuccessful. Classification accuracy requires a schema-valid answer with the correct category. NER computes per-document set intersections/differences and sums TP, FP and FN before calculating micro precision, recall and F1. Invalid output predicts an empty set; zero denominators score 0. Missing human references are never silently treated as empty gold.

`results/comparison.json` contains counts, scopes, per-type NER and a 5x6 confusion matrix (last column: invalid/pending). `comparison.csv` is the compact side-by-side table. Pending runs or incomplete gold are visibly marked. No classification F1 is claimed; task metrics are classification accuracy and entity micro-F1. `ready_for_final_report=true` requires complete returned responses, full reviewed gold and quantization metadata; it does not imply high accuracy or an already updated PDF.

Report generation is outside this project's runtime. Re-running `evaluate` updates metrics, not the static `report.pdf`. The included PDF matches the current reviewed gold and saved 200 responses; if either changes, update the report as well.

## Interpretation and limitations

Schema validity does not establish semantic correctness. Compare actual classification accuracy and NER F1 before concluding that constraints helped the task. The first three inputs informed the earlier prompt/repetition correction; this is a development-set evaluation, not an untouched blind test. Old news, truncation, a balanced small sample, one annotator and unknown pretraining overlap limit generalization. On the 95 articles with schema-valid prompt-only responses, both modes produce identical entity sets. The NER gain comes from five previously invalid outputs becoming eligible for scoring under the full-schema gate: 28 additional true positives and 5 false positives. This does not establish a broad improvement in semantic understanding.

## References

- Course specification: https://uorhan.github.io/files/NLP-Project-1/
- Corpus/source and provider citation: https://github.com/derekgreene/bbc-datasets ; Greene & Cunningham (2006), Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering, ICML. Full details in data/PROVENANCE.md.
- Meta model card: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- Ollama structured output: https://docs.ollama.com/capabilities/structured-outputs
- Generation parameters: https://docs.ollama.com/modelfile
- Python client: https://github.com/ollama/ollama-python
- Python JSON: https://docs.python.org/3/library/json.html
- Validator: https://python-jsonschema.readthedocs.io/en/stable/
- Metric definitions: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html

ChatGPT assisted with implementation/refactoring, annotation suggestions, checks and documentation. Berkay Doyuran ran the local inference and confirmed the reference review. Offline software checks are not model performance measurements.

Additional annotation-support reference: https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.8.0 (automatic drafts only, not an extra project runtime dependency).
