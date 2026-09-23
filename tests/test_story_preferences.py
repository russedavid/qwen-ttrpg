import copy

import pytest

from qwen_ttrpg.story_preferences import token_rows, validate_pairs
from qwen_ttrpg.util import digest, packed


def pair(group, split):
    row = {"id":group, "prompt":[{"role":"user", "content":"What does the ferryman say?"}],
           "chosen":"He names his price.", "rejected":"He says something about a price.",
           "provenance":{"group":group,"source_identity":group,"split":split}}
    row['review']={'verdict':'prefer_chosen','origin':'model','reviewer':'editor','chosen_valid':True,
                   'reason':'A specific usable response.',
                   'content_sha256':digest(packed({k:row[k]for k in ('prompt','chosen','rejected','provenance')}))}
    return row


@pytest.mark.parametrize('change',['test','group','source','target','review','duplicate'])
def test_preferences_reject_leakage_and_stale_reviews(change):
    a,b=pair('a','train'),pair('b','validation')
    if change=='test':b['provenance']['split']='test'
    elif change=='group':b['provenance']['group']='a'
    elif change=='source':b['provenance']['source_identity']='a'
    elif change=='target':a['chosen']='A different response.'
    elif change=='review':a['review']['chosen_valid']=False
    elif change=='duplicate':b=copy.deepcopy(a)
    with pytest.raises(ValueError):validate_pairs({'pairs':[a,b]})


class Tokenizer:
    eos_token_id=999
    def apply_chat_template(self,messages,tokenize=True,return_dict=False,add_generation_prompt=False,enable_thinking=True):
        assert enable_thinking is False
        prefix=[1,2,3]
        return prefix if add_generation_prompt else prefix+[ord(c)for c in messages[-1]['content']]+[999,10]


def test_native_completion_boundary_keeps_end_token_and_no_truncation():
    rows=[pair('a','train')]
    records,audits=token_rows(rows,Tokenizer(),100)
    assert records[0]['chat_template_kwargs']=={'enable_thinking':False}
    assert records[0]['chosen'][0]['content']==rows[0]['chosen']
    assert audits[0]['completion_tokens']['chosen']==len(rows[0]['chosen'])+2
    with pytest.raises(ValueError,match='protected prompt prefix'):
        token_rows(rows,Tokenizer(),10)
