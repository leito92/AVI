from Knowledge import Dataset
from Module import *
import datetime
from nltk.translate.bleu_score import sentence_bleu


db = Dataset()
tc = db.getTweets('AmazonHelp')
mr = Retrieval(tc)
q = "payments account close"
rc = mr.getRC(q)
print(rc)
mg = Generation(rc, 400, 1000)
a = datetime.datetime.now().replace(microsecond=0)
rf = mg.getRF_modelA()
#rf = mg.getRF_modelB()
#rf = mg.getRF_modelC()
b = datetime.datetime.now().replace(microsecond=0)
print(b - a)
reference = rc
candidate = rf
score = sentence_bleu(reference, candidate)
print(score)