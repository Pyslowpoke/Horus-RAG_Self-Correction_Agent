import json,unittest
import httpx,openai
from src.generators.deepseek_transport import DeepSeekClient

class TransportTests(unittest.TestCase):
 def test_request_contract_and_usage(self):
  def handle(request):
   body=json.loads(request.content)
   self.assertEqual(body['thinking'],{'type':'disabled'})
   self.assertEqual(body['response_format'],{'type':'json_object'})
   return httpx.Response(200,json={'choices':[{'message':{'content':'{"claims":[]}'}}],'usage':{'prompt_tokens':3,'completion_tokens':4}})
  c=DeepSeekClient({'api_base':'https://api.deepseek.com','api_key':'fixture'},httpx.MockTransport(handle))
  r=c.create(model='test',messages=[],timeout=2,response_format={'type':'json_object'},extra_body={'thinking':{'type':'disabled'}})
  self.assertEqual(r.usage.completion_tokens,4)
 def test_stream_consumes_sse_and_closes(self):
  transport=httpx.MockTransport(lambda r:httpx.Response(200,content=b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n'))
  c=DeepSeekClient({'api_base':'https://api.deepseek.com','api_key':'fixture'},transport)
  stream=c.create(model='test',messages=[],timeout=2,stream=True)
  self.assertEqual(next(iter(stream)).choices[0].delta.content,'hello')
  stream.close();self.assertTrue(stream.client.is_closed)
 def test_remote_error_body_is_withheld(self):
  c=DeepSeekClient({'api_base':'https://api.deepseek.com','api_key':'fixture'},httpx.MockTransport(lambda r:httpx.Response(401,json={'secret':'do-not-log'})))
  with self.assertRaises(openai.APIStatusError) as caught:c.create(model='test',messages=[],timeout=2)
  self.assertNotIn('do-not-log',str(caught.exception))
