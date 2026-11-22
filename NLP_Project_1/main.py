"""Run: python main.py check | run | evaluate | live | import OLD_PROJECT_FOLDER."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import zipfile
from inference import MODEL, HOST, OPTIONS, MAX_BYTES, TIMEOUT, SCHEMA, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from inference import connect, infer, process_output
from evaluation import validate_gold, evaluate

ROOT = Path(__file__).resolve().parent
LEGACY_SOURCE_SHA = '5e09838c3352e0ec20882a174f94b0d2991f20623d8f1fda0d5a4d109814aaf3'
DATASET_SHA = 'a351031a62325b3d7ff13146eeb035c22feca8290913b5183d1ff3be35109495'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, value):
    """Replace only after a complete write; a stopped run retains its last checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def load_inputs():
    """Use the already selected 100 inputs; never resample during refactoring."""
    inputs = read_json(ROOT / 'data/evaluation_inputs.json')
    archive = ROOT / 'data/bbc-fulltext.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != DATASET_SHA:
        raise ValueError('BBC archive checksum differs')
    if len(inputs) != 100 or len({r['id'] for r in inputs}) != 100:
        raise ValueError('Expected the existing 100 unique inputs')
    if Counter(r['label'] for r in inputs) != dict.fromkeys(['business','entertainment','politics','sport','tech'],20):
        raise ValueError('Expected 20 articles per class')
    with zipfile.ZipFile(archive) as z:
        for row in inputs:
            original = z.read('bbc/' + row['id']).decode('utf-8').replace('\r\n','\n').strip()
            text = original.encode('utf-8')[:MAX_BYTES].decode('utf-8', errors='ignore')
            if (text != row['input_text'] or row['label'] != row['id'].split('/')[0]
                    or hashlib.sha256(text.encode('utf-8')).hexdigest() != row['input_sha256']):
                raise ValueError('Input/reference mismatch: ' + row['id'])
    return inputs


def settings(inputs):
    return {'model': MODEL, 'host': HOST, 'options': OPTIONS, 'max_bytes': MAX_BYTES,
            'timeout': TIMEOUT, 'system_prompt': SYSTEM_PROMPT,
            'user_prompt_template': USER_PROMPT_TEMPLATE, 'schema': SCHEMA,
            'inputs': [{'id': r['id'], 'sha256': r['input_sha256']} for r in inputs]}


def load_run(mode, inputs):
    path = ROOT / 'results' / (mode + '.json')
    if not path.exists():
        return {'mode': mode, 'settings': settings(inputs), 'metadata': {}, 'records': []}
    run = read_json(path)
    if run['mode'] != mode or run['settings'] != settings(inputs):
        raise ValueError('Saved run has different requests/inputs; preserve it and inspect before restarting')
    records = run['records']
    ids = [r['id'] for r in records]
    if len(set(ids)) != len(ids) or not set(ids) <= {r['id'] for r in inputs}:
        raise ValueError('Duplicate or unexpected result IDs')
    return run


def evaluate_all(inputs):
    gold = validate_gold(read_json(ROOT / 'data/gold_entities.json'), inputs)
    runs = {mode: load_run(mode, inputs) for mode in ['prompt', 'schema']}
    if all(r['records'] for r in runs.values()):
        for key in ['model_digest', 'ollama_version', 'quantization']:
            if runs['prompt']['metadata'].get(key) != runs['schema']['metadata'].get(key):
                raise ValueError('Model/runtime differ between the two runs: ' + key)
    scores = {mode: evaluate(run['records'], inputs, gold) for mode, run in runs.items()}
    ready = (len(gold) == len(inputs) and all(s['received'] == len(inputs) for s in scores.values())
             and all(r['metadata'].get('quantization') for r in runs.values()))
    summary = {'ready_for_final_report': bool(ready), 'gold_reviewed': len(gold),
               'modes': scores, 'note': 'Development-set evaluation. Partial results are not final scores.'}
    save_json(ROOT / 'results/comparison.json', summary)
    rows = [['Metric', 'Prompt-only', 'Schema-constrained']]
    for key in ['raw_json_rate','cleaned_json_rate','schema_rate','classification_accuracy']:
        rows.append([key] + [scores[m][key] if scores[m]['pending'] == 0 else 'PENDING' for m in runs])
    for key in ['precision', 'recall', 'f1']:
        rows.append(['entity_micro_' + key] + [scores[m]['entities']['micro'][key]
            if len(gold) == 100 and scores[m]['received'] == 100 else 'PENDING' for m in runs])
    with (ROOT / 'results/comparison.csv').open('w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)
    for row in rows:
        print(' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row))
    print(f'Human-reviewed gold: {len(gold)}/100; ready for final report: {bool(ready)}')
    return summary


def run_all(inputs):
    client = metadata = None
    for mode in ['prompt', 'schema']:
        run = load_run(mode, inputs)
        done = {r['id'] for r in run['records']}
        if len(done) == len(inputs):
            print(mode, 'already has 100 attempts; no model calls repeated.')
            continue
        if client is None:
            client, metadata = connect()
        if run['records'] and any(run['metadata'].get(k) != metadata.get(k)
                                 for k in ['model_digest','ollama_version','quantization']):
            raise ValueError('Model/runtime changed during this run')
        run['metadata'] = metadata
        run.setdefault('created_utc', datetime.now(timezone.utc).isoformat())
        for index, row in enumerate(inputs, 1):
            if row['id'] in done:
                continue
            answer = infer(client, row['input_text'], mode)
            result = process_output(answer['raw_output']) if answer['raw_output'] is not None else None
            run['records'].append({'id': row['id'], **answer, 'processing': result})
            save_json(ROOT / 'results' / (mode + '.json'), run)
            print(f'{mode} [{index}/100] {row["id"]} '
                  f'raw_json={(result or {}).get("raw_jsonloads_ok",False)} '
                  f'schema={(result or {}).get("schema_ok",False)} '
                  f'stop={answer["done_reason"]} error={answer["inference_error"]}', flush=True)
    evaluate_all(inputs)


def import_old(old, inputs):
    """Validate and repackage old outputs offline; do not change raw answers or call Ollama."""
    old = Path(old).resolve()
    if old == ROOT:
        raise ValueError('Choose the original project folder, not this refactored folder')
    pending = []
    for mode in ['prompt', 'schema']:
        folder = old / 'results' / (mode + '100')
        if not (folder / 'manifest.json').exists():
            print('No legacy main-run manifest:', folder)
            continue
        manifest = read_json(folder / 'manifest.json')
        spec = manifest['spec']
        expected_config = {'model': MODEL, 'host': HOST, 'mode': mode, **OPTIONS,
                           'max_bytes': MAX_BYTES, 'timeout': TIMEOUT}
        fingerprint = hashlib.sha256(json.dumps(spec,sort_keys=True).encode('utf-8')).hexdigest()
        if (spec['config'] != expected_config or spec['inputs'] != inputs
                or spec['prompt'] != SYSTEM_PROMPT or spec.get('user_prompt_template') != USER_PROMPT_TEMPLATE
                or spec['source_sha256'] != LEGACY_SOURCE_SHA or spec['dataset_sha256'] != DATASET_SHA
                or fingerprint != manifest['fingerprint']):
            raise ValueError('Legacy conditions are not the verified post-fix version; inspect before importing')
        records = [read_json(p) for p in sorted((folder/'records').glob('*.json'))]
        if any(r['fingerprint'] != fingerprint for r in records):
            raise ValueError('Legacy record provenance mismatch')
        # Recompute processing; keep each raw response and original record unchanged.
        evaluate(records, inputs, {})
        target = ROOT / 'results' / (mode + '.json')
        run = {'mode': mode, 'settings': settings(inputs), 'metadata': manifest['model_metadata'],
               'created_utc': manifest['created_utc'], 'records': records,
               'original_manifest': manifest, 'import_note': 'File organization changed; requests unchanged.'}
        if target.exists() and read_json(target) != run:
            raise ValueError('Destination already has a different run; no overwrite: ' + str(target))
        pending.append((target, run))
    source_gold = old / 'data/gold_entities.json'
    if source_gold.exists():
        payload = read_json(source_gold)
        previous = read_json(ROOT / 'data/gold_entities.json')
        existing_gold = validate_gold(previous, inputs)
        imported_gold = validate_gold(payload, inputs)
        if any(rid not in imported_gold or imported_gold[rid] != pairs for rid,pairs in existing_gold.items()):
            raise ValueError('Import conflicts with an existing reviewed annotation; preserve both copies')
        if previous != payload and any(item['entities'] or item.get('notes') or item.get('reviewer') for item in previous['items']):
            raise ValueError('Destination gold has annotation work; merge manually rather than overwrite')
        pending.append((ROOT / 'data/gold_entities.json', payload))
    # Validate the whole import before writing any destination file.
    for path, value in pending:
        save_json(path, value)
        print('Imported:', path.relative_to(ROOT))
    evaluate_all(inputs)


def live():
    samples = read_json(ROOT / 'data/live_samples.json')
    client, metadata = connect()
    folder = ROOT / 'results' / ('live_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True, exist_ok=False)
    output = {'kind': 'fresh_live_demo_not_main_benchmark', 'metadata': metadata,
              'settings': settings([]), 'samples': samples, 'temperatures': [0.0,1.0], 'records': []}
    for sample in samples:
        print('\nSOURCE:',sample['source'],'\nTEXT:',sample['text'],flush=True)
        for temperature in [0.0,1.0]:
            for mode in ['prompt','schema']:
                answer = infer(client, sample['text'], mode, temperature)
                check = process_output(answer['raw_output']) if answer['raw_output'] is not None else None
                output['records'].append({'id': sample['id'], 'mode': mode, 'temperature': temperature,
                                          **answer, 'processing': check})
                save_json(folder / 'live.json', output)
                print(mode, 'T=',temperature,'seconds=',round(answer['latency_seconds'],1),flush=True)
                print(answer['raw_output'] if answer['raw_output'] is not None else answer['inference_error'],flush=True)
                print('raw_json=',(check or {}).get('raw_jsonloads_ok'),
                      'schema=',(check or {}).get('schema_ok'),flush=True)
    print('Fresh live results:', folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check','run','evaluate','live','import'])
    parser.add_argument('old_project', nargs='?')
    args = parser.parse_args()
    inputs = load_inputs()
    if args.command == 'check':
        _, metadata = connect()
        print(json.dumps({k:v for k,v in metadata.items() if k != 'model_details'},indent=2))
    elif args.command == 'run':
        run_all(inputs)
    elif args.command == 'evaluate':
        evaluate_all(inputs)
    elif args.command == 'live':
        live()
    elif args.old_project:
        import_old(args.old_project, inputs)
    else:
        parser.error('import requires the original project folder path')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit('Interrupted. Saved attempts remain. Resume main runs with the same settings.')
    except Exception as error:
        raise SystemExit(f'{type(error).__name__}: {error}')
