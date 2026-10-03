"""Byte-for-byte native renderer parity with pinned, independent Jinja source.

The input is the NInfer normalized contract (object arguments, reasoning_content,
XML tools, no text truncation). Python/Jinja is a test dependency only.
"""
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
from jinja2.sandbox import ImmutableSandboxedEnvironment

renderer, source = sys.argv[1:]
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
env.filters['tojson'] = lambda x: json.dumps(x, ensure_ascii=False)
def fail(msg):
    raise ValueError(msg)
env.globals['raise_exception'] = fail
template = env.from_string(Path(source).read_text(encoding='utf-8'))
tool = {'type':'function','function':{'name':'test_tool','description':'quotes " <>& café','parameters':{'type':'object','properties':{'command':{'type':'string'}}}}}
def m(role, content='', **extra):
    return dict(role=role, content=content, **extra)
cases = []
def case(messages, options=None, tools=None):
    cases.append(dict(messages=messages,options=options or {},tools=tools or []))
base = [m('system',' header '),m('developer','policy'),m('user','task'),
        m('assistant','<think>\nold\n</think>\n\nbody',reasoning_content='old',tool_calls=[{'function':{'name':'test_tool','arguments':{'command':'a\nb"c','nested':{'a':[True,None,'雪']},'n':2.5}}}]),
        m('tool','{"error":"first"}'),m('assistant',''),m('tool','Failed to open'),
        m('developer','per-turn policy'),m('assistant','continue'),m('user','follow-up')]
for n in range(1,len(base)+1):
    for thinking, preserve, generation, effort in itertools.product((True,False),(True,False),(True,False),('low','medium','xhigh')):
        case(base[:n],dict(enable_thinking=thinking,preserve_thinking=preserve,add_generation_prompt=generation,reasoning_effort=effort),[tool])
for content in ['plain','<think>x</think>answer','<thinking>\nx\n</thinking>\nanswer','x\n</ think>answer','x\n</think >answer','</think>answer','</thinking>answer','Explain `<think>` and `</think>` in code.','<think>first</think>last</think>answer','', ' \n\t ', '\u00a0answer\u2003']:
    for reasoning in ('','explicit',' \n '):
        case([m('user','task'),m('assistant',content,reasoning_content=reasoning)],tools=[tool])
for result in ['Error: wrong','{"error":null}','{"error": false}','{"error":""}','{"error":"bad"}','{"status":"error"}','Process exited with code 0','Process exited with code 1','Exit code: 0','Exit code: 2','throw new Error("x")','console.error("x")','function f() { Error: }','import x\nError:','Traceback (most recent call last):','command not found','fatal: failure','$ command\nError:','took 1ms Error:','Error:'+'x'*600,'é'*115+'Error:','雪'*115+'Error:']:
    case([m('user','task'),m('tool',result),m('assistant','retry'),m('tool',result)],tools=[tool])
for tag in ('off','on','low','minimal','medium','high','xhigh','max','ultracode','extreme'):
    case([m('system','policy <|think_off|>'),m('user',[{'type':'text','text':'a <|think_'+tag+'|>'},{'type':'image'}]),m('tool','<|think_off|>')],dict(add_vision_id=True),[tool])
case([m('tool','Error: no user')],tools=[tool])
case([m('assistant','history')]*55,dict(preserve_thinking=False))
case([m('user',[{'type':'image'},{'type':'text','text':'compare'},{'type':'video'},{'type':'image'}])],dict(add_vision_id=True),[tool])
case([m('system',[{'type':'image'}]),m('user','x')])
case([])
case([m('unknown','x')])
rng = random.Random(225)
texts = ['ok','x\n</think>\ny','Error: invalid','{"error":null}','<|think_off|> x','<|think_xhigh|> y','<|think_on|><|think_low|>','a\nb','雪 café', '']
for _ in range(500):
    messages = [m(rng.choice(['user','assistant','tool','system','developer']),rng.choice(texts)) for _ in range(rng.randrange(1,12))]
    case(messages, dict(enable_thinking=rng.choice([True,False]),preserve_thinking=rng.choice([True,False])),[tool] if rng.randrange(2) else [])
payload = ''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases)
proc = subprocess.run([renderer,source],input=payload,text=True,encoding='utf-8',capture_output=True,check=True)
lines = proc.stdout.splitlines()
assert len(lines) == len(cases), (len(lines),len(cases),proc.stderr)
failures = []
for i,(c,line) in enumerate(zip(cases,lines)):
    actual = json.loads(line)
    try:
        expected = template.render(messages=c['messages'],tools=c['tools'],**{'add_generation_prompt':True, **c['options']})
    except Exception:
        if 'error' not in actual: failures.append((i,'expected error'))
        continue
    if actual.get('text') != expected:
        got = actual.get('text','ERROR: '+actual.get('error',''))
        at = next((j for j,(a,b) in enumerate(zip(got,expected)) if a!=b),min(len(got),len(expected)))
        failures.append((i,dict(at=at,actual=got[max(0,at-60):at+160],expected=expected[max(0,at-60):at+160],input=c)))
    if actual.get('rewrite') is not None:
        prefix = actual['text'].encode('utf-8')[:actual['rewrite']]
        assert prefix.endswith(b'<|im_start|>assistant\n'), (i,'invalid prefix rewrite boundary')
    assert actual.get('starts_in_reasoning') == expected.endswith('<|im_start|>assistant\n<think>\n'), (i,'incorrect output channel state')
if failures:
    print(json.dumps(failures[:8],ensure_ascii=False,indent=2))
    raise SystemExit(f'{len(failures)}/{len(cases)} parity cases failed')
print(f'{len(cases)} native/Jinja parity cases passed; rewrite offsets validated.')
