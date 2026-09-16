"""Cheap contextual search queries and cache keys that account for conversation state."""
from datetime import datetime, timedelta
import os
import re
import requests
from src.config import fingerprint
from src.runtime import bounded_timeout, check_budget, timed


def replace_relative_dates(text, now=None):
    now = now or datetime.now()
    offsets = {'大前天': -3, '前天': -2, '昨天': -1, '今天': 0, '大后天': 3, '后天': 2, '明天': 1}
    return re.sub('|'.join(sorted(offsets, key=len, reverse=True)),
                  lambda match: (now + timedelta(days=offsets[match[0]])).strftime('%Y年%m月%d日'), text)


def contextual_query(query, history):
    # Keep the user question intact; only add the latest user turn for explicit references.
    if not any(word in query for word in ('它', '他', '她', '这个', '那个', '上述', '刚才', '继续')):
        return query
    previous = next((msg['content'] for msg in reversed(history) if msg['role'] == 'user'), '')
    return f'{previous[:160]} {query}' if previous else query


def cache_key_for(query, mode, history, preferences, version, config):
    return fingerprint({'query': query, 'mode': mode, 'history': history,
                        'preferences': preferences, 'index': version, 'config': config})


def rewrite_query(query, timeout=8):
    token = os.getenv('BAIDU_API_KEY')
    if not token:
        return query
    with timed('query_rewrite'):
        try:
            response = requests.post('https://qianfan.baidubce.com/v2/tools/query_rewrite',
                headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'},
                json={'query': query}, timeout=bounded_timeout(timeout))
            response.raise_for_status()
            data = response.json()
            result = data.get('altered_query') if data.get('code') == 0 else None
            check_budget()
            return result if isinstance(result, str) and result.strip() else query
        except (requests.RequestException, ValueError):
            return query
