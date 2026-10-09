"""Direct HTTP transport shared in behavior with the working workbench path.

No hidden retries; closes streams and withholds remote error bodies.
"""
import json
from types import SimpleNamespace
import httpx
import openai


def objectify(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: objectify(v) for k, v in value.items()})
    if isinstance(value, list):
        return [objectify(v) for v in value]
    return value


class EventStream:
    def __init__(self, client, response):
        self.client, self.response = client, response

    def __iter__(self):
        try:
            for line in self.response.iter_lines():
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    break
                if data:
                    yield objectify(json.loads(data))
        except httpx.RequestError as error:
            raise openai.APIConnectionError(request=error.request) from error

    def close(self):
        self.response.close()
        self.client.close()


class DeepSeekClient:
    def __init__(self, config, transport=None):
        self.config = config
        self.transport = transport
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        timeout = kwargs.pop('timeout')
        extra = kwargs.pop('extra_body', {})
        payload = {**kwargs, **extra}
        client = httpx.Client(timeout=httpx.Timeout(timeout, connect=min(timeout, 3), write=min(timeout, 5), pool=min(timeout, 3)), transport=self.transport)
        try:
            request = client.build_request('POST', self.config['api_base'].rstrip('/') + '/chat/completions',
                headers={'Authorization': 'Bearer ' + self.config['api_key']}, json=payload)
            response = client.send(request, stream=bool(payload.get('stream')))
            if response.is_error:
                raise openai.APIStatusError(f'Model service HTTP {response.status_code}', response=response, body=None)
            if payload.get('stream'):
                return EventStream(client, response)
            return objectify(response.json())
        except httpx.RequestError as error:
            raise openai.APIConnectionError(request=error.request) from error
        finally:
            if not payload.get('stream'):
                client.close()
            elif 'response' not in locals() or response.is_error:
                client.close()
