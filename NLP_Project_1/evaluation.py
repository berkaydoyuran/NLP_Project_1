"""Metrics recomputed from original responses and manually reviewed reference labels."""
import hashlib
from inference import LABELS, process_output

TYPES = {'PERSON', 'ORG', 'LOC'}

def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def validate_gold(payload, inputs, require_all=False):
    if payload.get('protocol') != 'exact_surface_type_sets_v1':
        raise ValueError('Unsupported annotation protocol')
    expected = {r['id']: r for r in inputs}
    seen, gold = set(), {}
    for item in payload.get('items', []):
        rid = item['id']
        if rid not in expected or rid in seen:
            raise ValueError('Duplicate/unexpected annotation id: ' + rid)
        seen.add(rid)
        row = expected[rid]
        if item['input_sha256'] != row['input_sha256'] or text_hash(item['input_text']) != row['input_sha256']:
            raise ValueError('Annotation text differs from model input: ' + rid)
        if not isinstance(item.get('reviewed'), bool):
            raise ValueError('reviewed must be a boolean')
        pairs = set()
        for entity in item['entities']:
            if set(entity) != {'text','type'} or entity['type'] not in TYPES:
                raise ValueError('Invalid entity fields/type: ' + rid)
            if not entity['text'].strip() or entity['text'] not in row['input_text']:
                raise ValueError('Entity is not an exact input substring: ' + rid)
            pair = (entity['text'], entity['type'])
            if pair in pairs:
                raise ValueError('Duplicate gold entity: ' + rid)
            pairs.add(pair)
        if item['reviewed']:
            if not item.get('reviewer', '').strip() or not item.get('reviewed_at', '').strip():
                raise ValueError('Reviewed annotations need a reviewer and date: ' + rid)
            gold[rid] = pairs
    if seen != set(expected):
        raise ValueError('Annotation file must contain exactly the evaluation inputs')
    if require_all and len(gold) != len(expected):
        raise ValueError(f'Manual review incomplete: {len(gold)}/{len(expected)}')
    return gold

def entity_metrics(records, inputs, gold):
    """Micro exact (surface,type) set F1; invalid outputs predict the empty set."""
    by_id = {r['id']: r for r in records}
    counts = dict(tp=0, fp=0, fn=0)
    per_type = {t:dict(tp=0,fp=0,fn=0) for t in sorted(TYPES)}
    examples = []
    for row in inputs:
        if row['id'] not in gold:
            continue
        expected = gold[row['id']]
        record = by_id.get(row['id'], {})
        check = record.get('processing') or {}
        predicted = {(e['text'],e['type']) for e in check['parsed']['entities']} if check.get('schema_ok') else set()
        tp, fp, fn = predicted & expected, predicted - expected, expected - predicted
        for name, pairs in [('tp',tp),('fp',fp),('fn',fn)]:
            counts[name] += len(pairs)
            for _, entity_type in pairs:
                per_type[entity_type][name] += 1
        if fp or fn:
            examples.append({'id':row['id'],'false_positives':sorted(fp),'false_negatives':sorted(fn)})
    def scores(c):
        tp,fp,fn = c['tp'],c['fp'],c['fn']
        # Explicit zero_division=0 convention, even when both sets are empty.
        return {**c,'precision':tp/(tp+fp) if tp+fp else 0.0,
                'recall':tp/(tp+fn) if tp+fn else 0.0,
                'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}
    return {'reviewed_samples':len(gold), 'scope':'all_inputs' if len(gold)==len(inputs) else 'reviewed_subset_only',
            'micro':scores(counts) if gold else None,
            'per_type':{t:scores(c) for t,c in per_type.items()} if gold else None,
            'errors':examples,'unit':'unique (exact case-sensitive surface,type) per document; not mention offsets',
            'zero_division':0}


def evaluate(records, inputs, gold):
    """All-input denominators retain invalid, failed and missing responses."""
    expected = {row['id']: row for row in inputs}
    if len({r['id'] for r in records}) != len(records) or any(r['id'] not in expected for r in records):
        raise ValueError('Duplicate or unexpected result IDs')
    checked = [{**r, 'processing': process_output(r['raw_output']) if r['raw_output'] is not None else None}
               for r in records]
    n = len(inputs)
    raw = cleaned = valid = correct = received = 0
    matrix = [[0] * 6 for _ in LABELS]
    by_id = {r['id']: r for r in checked}
    for row in inputs:
        record = by_id.get(row['id'])
        result = (record or {}).get('processing') or {}
        received += bool(record and record['raw_output'] is not None)
        raw += bool(result.get('raw_jsonloads_ok'))
        cleaned += bool(result.get('parse_ok'))
        valid += bool(result.get('schema_ok'))
        prediction = result['parsed']['category'] if result.get('schema_ok') else None
        correct += prediction == row['label']
        matrix[LABELS.index(row['label'])][LABELS.index(prediction) if prediction else 5] += 1
    return {'planned': n, 'attempted': len(records), 'received': received,
            'pending': n - len(records), 'inference_failures': len(records) - received,
            'raw_json_rate': raw / n, 'cleaned_json_rate': cleaned / n,
            'schema_rate': valid / n, 'classification_accuracy': correct / n,
            'length_limited': sum(r.get('done_reason') == 'length' for r in records),
            'confusion_matrix': matrix, 'confusion_rows': LABELS,
            'confusion_columns': LABELS + ['invalid_or_pending'],
            'entities': entity_metrics(checked, inputs, gold)}
