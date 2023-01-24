import json
from tqdm import tqdm
from seal import FMIndex, SEALSearcher
from seal.evaluate import evaluator

searcher = SEALSearcher.load("NQ_320k/FM_Index/NQ_320k.base.fm_index", "checkpoints/checkpoint_best.pt", device='cuda:0' )
searcher.include_keys = True
myevaluator = evaluator()

query_list = []
result = []
truth = []

with open("NQ_320k/dev4retrieval.json", "r") as f:
    for i, line in enumerate(tqdm(f)):
        line = json.loads(line)
        tmp = []
        truth.append([line['docid']])
        for doc in searcher.search(line['query'], k=100):
            tmp.append(doc.docid)
        result.append(tmp)

res = myevaluator.evaluate_ranking(truth, result)
print(f"mrr@5:{res['mrr5']}, mrr@10:{res['mrr10']}, mrr:{res['mrr']}, p@1:{res['p1']}, p@10:{res['p10']}, p@20:{res['p20']}, p@100:{res['p100']}, r@1:{res['r1']}, r@5:{res['r5']}, r@10:{res['r10']}, r@100:{res['r100']}")