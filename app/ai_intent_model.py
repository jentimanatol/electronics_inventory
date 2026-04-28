import math, re
from collections import Counter, defaultdict

INTENT_EXAMPLES={
 'show_all':['show all items','list all items','show everything in inventory','what do I have in stock','full inventory list','display all records'],
 'show_categories':['show categories','list all categories','group items by category','show inventory types','what categories do I have','summarize by type'],
 'search':['do I have resistors','do I have resistor in store','do we have any sensors','find capacitors','show Arduino boards','find stepper motor drivers','where are my tools'],
 'low_stock':['what should I restock','show low stock items','what has low quantity','what should I buy','shortage warning','items with qty low'],
 'duplicates':['show duplicate items','do I have duplicate capacitors','find repeated records','same resistor listed twice','duplicate check'],
 'inventory_value':['what is the total value','show inventory price','how much money is in inventory','what is the cost of my items','inventory worth','total budget value'],
 'help':['help','what can I ask','examples','how to use AI assistant'],
}
STOPWORDS={'a','an','the','i','we','my','your','you','do','does','is','are','in','on','of','for','to','with','any','please','can','me'}
SYNONYMS={'show':{'display','list','print'},'all':{'everything','full','complete'},'category':{'group','type','kind'},'search':{'find','lookup','where'},'price':{'cost','value','worth','money','budget'},'low':{'restock','shortage','missing','buy','quantity','qty'},'duplicate':{'same','repeated','twice'},'resistor':{'resistance','ohm','kohm'},'capacitor':{'cap','uf','nf','pf'},'sensor':{'temperature','humidity','moisture','hall'}}
IRREGULAR={'categories':'category','batteries':'battery','supplies':'supply','boxes':'box','switches':'switch','wires':'wire','tools':'tool','materials':'material','documents':'document','books':'book','sensors':'sensor','resistors':'resistor','capacitors':'capacitor','modules':'module','connectors':'connector','motors':'motor','drivers':'driver','boards':'board','items':'item','types':'type'}

def singularize(word):
    if word in IRREGULAR: return IRREGULAR[word]
    if len(word)>4 and word.endswith('ies'): return word[:-3]+'y'
    if len(word)>4 and word.endswith('es') and not word.endswith(('ses','xes')): return word[:-2]
    if len(word)>3 and word.endswith('s') and not word.endswith(('ss','us')): return word[:-1]
    return word

def tokenize(text):
    out=[]
    for w in re.findall(r'[a-zA-Z0-9]+',(text or '').lower()):
        for t in {w,singularize(w)}:
            if len(t)>1 and t not in STOPWORDS:
                out.append(t); out.extend(SYNONYMS.get(t,set()))
    return out

class TinyIntentModel:
    def __init__(self,examples):
        self.intent_counts=Counter(); self.token_counts=defaultdict(Counter); self.vocab=set(); self.total=0
        for intent,phrases in examples.items():
            for phrase in phrases:
                self.total+=1; self.intent_counts[intent]+=1
                c=Counter(tokenize(phrase)); self.token_counts[intent].update(c); self.vocab.update(c)
    def predict(self,question):
        toks=tokenize(question)
        if not toks: return {'intent':'help','confidence':1.0,'tokens':[]}
        V=max(len(self.vocab),1); scores={}
        for intent in self.intent_counts:
            score=math.log(self.intent_counts[intent]/self.total)
            total=sum(self.token_counts[intent].values())+V
            for tok in toks: score+=math.log((self.token_counts[intent][tok]+1)/total)
            scores[intent]=score
        m=max(scores.values()); exp={k:math.exp(v-m) for k,v in scores.items()}; den=sum(exp.values()) or 1.0
        ranked=sorted(exp.items(), key=lambda kv: kv[1], reverse=True)
        return {'intent':ranked[0][0],'confidence':round(ranked[0][1]/den,3),'tokens':toks,'ranked':[(k,round(v/den,3)) for k,v in ranked[:3]]}
MODEL=TinyIntentModel(INTENT_EXAMPLES)
def classify_inventory_question(question): return MODEL.predict(question)
