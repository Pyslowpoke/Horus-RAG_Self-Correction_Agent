"""Deterministic reference integrity; deliberately not a semantic fact checker."""
import re


def audit_citations(answer, evidence_count, applicable=True):
    if not applicable or not evidence_count:
        return {'status': 'not_applicable', 'used_ids': [], 'invalid_ids': []}
    # Support [1], [1, 2], [1，2], [1、2]. Ignore Markdown link labels
    # and fenced code so examples and link text are not treated as references.
    text = re.sub(r'```.*?```', '', answer, flags=re.S)
    matches = re.findall(r'\[(\d+(?:\s*[,，、]\s*\d+)*)\](?!\()', text)
    used = sorted({int(number) for match in matches for number in re.findall(r'\d+', match)})
    invalid = [number for number in used if not 1 <= number <= evidence_count]
    return {'status': 'invalid' if invalid else ('valid' if used else 'missing'),
            'used_ids': used, 'invalid_ids': invalid}
