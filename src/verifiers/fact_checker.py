"""Structured verification. Graph owns the rewrite loop; this class checks once."""
import json
import logging
import re
from src.generators.prompt_templates import FACT_CHECK_PROMPT, REWRITE_RESPONSE_PROMPT

logger = logging.getLogger(__name__)


class VerificationError(ValueError):
    pass


class FactChecker:
    def __init__(self, llm_client, retriever=None, max_retries=1, system_identity=''):
        self.llm_client = llm_client
        self.max_retries = max_retries

    def _parse_json_response(self, raw):
        raw = (raw or '').strip()
        if raw.startswith('```'):
            raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw)
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as error:
            raise VerificationError('核查结果不是有效 JSON') from error
        if isinstance(parsed, dict):
            parsed = parsed.get('claims')
        if not isinstance(parsed, list):
            raise VerificationError('核查结果必须是陈述列表')
        return parsed

    def _validate_log_entry(self, entry, context):
        if not isinstance(entry, dict):
            raise VerificationError('核查条目必须是对象')
        claim = entry.get('claim') or entry.get('statement')
        verdict = entry.get('verdict')
        evidence = entry.get('evidence', '')
        if not isinstance(claim, str) or not claim or verdict not in ('支持', '矛盾', '证据不足') or not isinstance(evidence, str):
            raise VerificationError('核查条目字段不完整')
        # A fabricated quote is not evidence, even if the judge labels it supported.
        if verdict in ('支持', '矛盾') and (not evidence.strip() or evidence.strip() not in context):
            verdict = '证据不足'
        return {'claim': claim, 'verdict': verdict, 'evidence': evidence}

    def check_once(self, answer, context):
        raw = self.llm_client.generate([{'role': 'user', 'content':
            FACT_CHECK_PROMPT.format(response=answer, context=context)}])
        return [self._validate_log_entry(entry, context) for entry in self._parse_json_response(raw)]

    def _rewrite(self, answer, failed_claims, context):
        return self.llm_client.generate([{'role': 'user', 'content': REWRITE_RESPONSE_PROMPT.format(
            original_response=answer, failed_claims=json.dumps(failed_claims, ensure_ascii=False), context=context)}])

    def execute_pipeline(self, query, initial_answer, context):
        """Compatibility API for callers outside the graph, with a single bounded loop."""
        answer, history = initial_answer, []
        for attempt in range(self.max_retries + 1):
            checks = self.check_once(answer, context)
            history.extend({**entry, 'round': attempt} for entry in checks)
            failed = [entry for entry in checks if entry['verdict'] != '支持']
            if not failed or attempt == self.max_retries:
                return {'final_answer': answer, 'verification_log': checks, 'verification_history': history,
                        'failed_claims': failed, 'retry_count': attempt}
            answer = self._rewrite(answer, failed, context)
