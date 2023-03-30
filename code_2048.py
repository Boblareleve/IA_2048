import random as r
import numpy as np
import csv
import os

def make(x, y, m) :
    rand_list=[]
    for i in range(x*y):
        rand_list.append(r.random()*m*2-m)
    return [rand_list,[x,y]]

def weight_and_biais_csv(FILE):
    out = []

    var = 0.5
    out+=make(16,64, var) # W1
    out+=make(64,1, var) # B1
    #out+=make(64,64, var)
    #out+=make(64,1, var)
    out+=make(64,16, var)
    out+=make(16,1, var)
    out+=make(16,4, var)
    out+=make(4,1, var)

    with open(FILE, 'w') as csvfile:
        csv.writer(csvfile,lineterminator='\n').writerows(out)

def weight_and_biais_array(var=0.1):
    return [
    np.random.rand(16,64)*var*2-var,
    np.random.rand(1,64)*var*2-var,
    np.random.rand(64,16)*var*2-var,
    np.random.rand(1,16)*var*2-var,
    np.random.rand(16,4)*var*2-var,
    np.random.rand(1,4)*var*2-var,
    ]

def get_weight_and_biais_csv(FILE): # i%4=1 : value weight ; i%4=2 : shape weight ; i%4=3 value biais ; i%4=0 shape biais
    out = []
    with open(FILE, "r") as csvfile:
        weight_biais = csv.reader(csvfile)
        for i in weight_biais:
               out.append(i)

    d=[]
    i=0
    while len(out)>i : 
        d.append(
        np.asarray(out[i]).astype(float).reshape((int(out[i+1][0]),int(out[i+1][1])))
        ) #weight
        d.append(
        np.asarray(out[i+2]).astype(float).reshape((int(out[i+3][0])))
        ) #biais
        i+=4
    return d

def mod_weight_and_biais_csv(FILE_b, FILE_prod):
    result = get_weight_and_biais_csv(FILE_b)
    result_c = []
    for i in result:
        result_c.append(i.ravel()) # list()
        result_c.append(i.shape)
    out = []
    var = 0.05
    out+=make(16,64, var) # W1
    out+=make(64,1, var) # B1
    #out+=make(64,64, var)
    #out+=make(64,1, var)
    out+=make(64,16, var)
    out+=make(16,1, var)
    out+=make(16,4, var)
    out+=make(4,1, var)

    i = 0
    while i<len(result_c) :
        result_c[i]=result_c[i] + np.array(out[i])
        i+=2

    with open(FILE_prod, "w") as csvfile:
        csv.writer(csvfile,lineterminator='\n').writerows(result_c)

def set_weight_and_biais_csv(model, FILE_prod):
    result_c = []
    for i in model :
        result_c.append(i.ravel())
        result_c.append(i.shape)
    with open(FILE_prod, "w") as csvfile:
        csv.writer(csvfile,lineterminator='\n').writerows(result_c)

def mod_weight_and_biais_array(array_b):
    out = weight_and_biais_array(0.01)
    i = 0
    while i<len(array_b) :
        out[i]=out[i] + array_b[i]
        i+=1
    return out

def sigmoide(Z):
    return np.where(Z > 0, Z, Z * 0.01)
    # return 1.0/(1+(np.exp(-Z)))

def L_computing(p_L,W,B):# previous Layer ; weight set ; biais set
    return (p_L @ W)+B #calcule de tout les neurones du Layer

def IA(IA_t, tryed, model_used):
    IA_t = np.log2(IA_t, where=IA_t>0).astype('int32')

    IA_t=np.ravel(IA_t)
    i=0
    while len(model_used)>i :
        IA_t=L_computing(IA_t, model_used[i],model_used[i+1])
        IA_t=sigmoide(IA_t)
        i+=2
    IA_t-=tryed*1000
    return int(np.argmax(IA_t)) # ligne avec l'actions pour chaque partie

def entrer(m):
    l={"z":0,"s":2,"q":3,"d":1}
    x=input(m).lower()
    if x not in ("z","s","q","d"):
        x="z"
    return l[x[len(x)-1]]

def add_random(t_add_random, alive_r):
    if alive_r : 
        li=[]
        for l in range(4):
            for c in range(4):
                if t_add_random[l,c]==0 :
                    li.append((l,c))
        t_add_random[r.choice(li)] = r.choices([2,4],weights=[0.9,0.1])[0]
    return t_add_random

def slide(t_slide,d): 
    out=np.rot90(t_slide,k=d)
    mouv = 0
    
    for l in range(4):
        for c in range(4):
            if out[l,c]!=0:
                up=0
                if l>0  :
                    if out[l-up-1,c]==0 :
                        up+=1
                while l-up>0 and out[l-up-1,c]==0 :
                    up+=1
                if l-up>0 and out[l-up-1,c]==out[l,c] :
                    out[l-up-1,c]=out[l,c]*2
                    out[l,c]=0
                    mouv+=1
                else :
                    out[l-up,c]=out[l,c]
                    if up!=0:
                        out[l,c]=0
                        mouv+=1
    out=np.rot90(out,k=4-d)
    return out, bool(mouv)

def to_loop(t_loop,player, alive_l, turn_l, model_l):

    defeat_list=np.zeros((4))
    state = False
    Input=int()
    while not state :
        if player :
            Input=entrer("mouv :")
        else : 
            Input=IA(t_loop, defeat_list, model_l)
        t_loop, state = slide(t_loop, Input)
        if not state :
            defeat_list[Input]=Input+1
            if sum(defeat_list)==10:
                alive_l = False
                state = True
    
    turn_l+=1
    return t_loop, Input, alive_l, turn_l

def score_compt(t_score):
    return np.sum(t_score)

def main(model_m):   
    alive = True
    turn = 0
    fild = np.zeros((4,4), dtype="int16")
    Player = False
    
    add_random(fild, alive)
    #print("-"*15,"\nturn= ",turn,"\n",fild.astype(int),"\n")
    while alive :
        fild, mouv, alive, turn = to_loop(fild,Player, alive, turn, model_m) # True : h ; False IA
        add_random(fild, alive)
        #print("-"*15,"\nturn= ",turn,"--", mouv,"\n",fild.astype(int),"\n")
    score = score_compt(fild)
    print(f"You lose with a score of {score}pt")
    return score

def training_fast(model=False,times=100,num_keep=10) : # model (array) # times nombre de sortie totale
    if not model :
        model = []
        for i in range(times) :
            model.append(weight_and_biais_array(var=1.0))
       
    score = []
    for i in model : 
        score.append(main(i))

    keept = sorted(range(len(score)), key=lambda i: score[i], reverse=True)[:num_keep]

    son_model = []
    i=0
    while i < (times/num_keep) : 
        for k in keept : 
            son_model.append(mod_weight_and_biais_array(model[k]))
        i+=1
    return score, son_model

score=[]
avrage_score=[]
model=[]
temp=()
temp = training_fast()
score.append(temp[0])
avrage_score.append(sum(temp[0])/len(temp[0]))
model.append(temp[1])
for i in range(0) :
    temp=()
    temp = training_fast(model[-1])
    avrage_score.append(sum(temp[0])/len(temp[0]))
    score.append(temp[0])
    model.append(temp[1])

meta_model = 0
meta_model_num = 0
model_num = 0

directory = f'model_{meta_model}.{meta_model_num}'
os.mkdir(directory)
for i in model : 
    for k in i : 
        result_c = []
        for j in k:
            result_c.append(j.ravel()) # list()
            result_c.append(j.shape)
        with open(f'./{directory}/WB{model_num}.csv', "w") as csvfile:
            csv.writer(csvfile,lineterminator='\n').writerows(result_c)
        model_num+=1

