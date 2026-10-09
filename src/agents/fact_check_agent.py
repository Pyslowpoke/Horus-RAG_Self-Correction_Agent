"""Single verification pass; never treat malformed output as success."""
import logging
from src.runtime import RequestTimeout
logger = logging.getLogger(__name__)


def make_fact_check_agent(fact_checker):
    def fact_check_agent(state):
        if (state.get('search_mode') == 'self_aware' or not state.get('all_docs')
                or not state.get('answer') or state.get('generation_error')):
            return {'verification_log': [], 'failed_claims': [], 'verification_status': 'skipped'}
        if any(marker in state['answer'] for marker in ('无法回答', '不能回答', '缺少依据', '没有足够依据')):
            return {'verification_log': [], 'failed_claims': [], 'verification_status': 'unanswered'}
        try:
            checks = fact_checker.check_once(state['answer'], state['context'])
            failed = [entry for entry in checks if entry['verdict'] != '支持']
            history = state.get('verification_history', []) + [
                {**entry, 'round': state.get('retry_count', 0)} for entry in checks]
            return {'verification_log': checks, 'verification_history': history,
                    'failed_claims': failed,
                    'verification_status': 'failed' if failed else ('passed' if checks else 'no_claims')}
        except RequestTimeout:
            raise
        except Exception:
            logger.exception('事实核查失败')
            return {'verification_log': [], 'failed_claims': [], 'verification_status': 'error'}
    return fact_check_agent
