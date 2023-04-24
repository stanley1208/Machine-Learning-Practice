import numpy as np

tag2id,id2tag={},{}
word2id,id2word={},{}

with open('WSJ_02-21.pos','r') as f:
    items=f.readlines()
    word,tag=items[0],items[1].strip()

    if word not in word2id:
        word2id[word]=len(word2id)
        id2word[len(id2word)]=word
    if tag not in tag2id:
        tag2id[tag]=len(tag2id)
        id2tag[len(id2tag)]=tag

M=len(word2id)  # M: 辭典的大小
N=len(tag2id)   # N: 詞性的種類個數

# print(M,N)
# print(word2id)
# print(id2word)
# print(tag2id)
# print(id2tag)

# 構建 pi, A, B
pi=np.zeros(N) # 每個詞性出現在句子第一個位置的機率  N: # of tags
A=np.zeros((N,M)) # A[i][j]: 給定tag i, 出現單辭j的機率。 N: # of tags M: # of words in dictionary
B=np.zeros((N,N)) # B[i][j]: 之前的狀態是i, 之後轉換成狀態j的機率 N: # of tags

prev_tag=""
with open('WSJ_02-21.pos','r') as f:
    items=f.readlines()
    wordId,tagId=word2id[items[0]],tag2id[items[1].strip()]
    if prev_tag=="": # 代表著句子的開始
        pi[tagId]+=1
        A[tagId][wordId]+=1
    else:   # 不是句子的開始
        A[tagId][wordId]+=1
        B[tag2id[prev_tag]][tagId]+=1

    if items[0]==".":
        prev_tag=""
    else:
        prev_tag=items[1].strip()

# print(pi)
# print(id2tag)
# print(A[0])
# print(word2id)
# print(B)
# print(id2tag)

pi=pi/sum(pi)

for i in range(N):
    A[i] /= sum(A[i])
    B[i] /= sum(B[i])

print(pi)
print(A)
print(B)

def log(v):
    if v==0:
        return np.log(v+0.00000000001)

def viterbi(x,pi,A,B):
    '''
    :param x: user input string/sentence: x: "I like playing soccer"
    :param pi: initial probability of tags
    :param A: 給定tag, 每個單詞出現的機率
    :param B: tag之間的轉移機率
    :return:
    '''
    x=[word2id[word] for word in x.split(" ")] # x: [4521, 412, 542 ...]
    T=len(x)

    dp=np.zeros((T,N))  # dp[i][j]: w1...wi, 假設wi的tag是第j個tag
    ptr=np.array([[0 for x in range(N)] for y in range(T)])

    for j in range(N):  # basecase for dp
        dp[0][j]=log(pi[j])+log(A[j][x[0]])

    for i in range(1,T): # 每個單詞
        for j in range(N): # 每個詞性
            dp[i][j]=-np.inf
            for k in range(N): # 從每一個k可以到達j
                score=dp[i-1][k]+log(B[k][j])+log(A[j][x[i]])
                if score > dp[i][j]:
                    dp[i][j]=score
                    ptr[i][j]=k


    # decoding: 把最好的 tag sequence 打印出來
    best_seq=[0]*T # best_seq=[1,5,2,23,4,...]
    # step1: 找出對應於最後一個單詞的詞性
    best_seq[T-1]=np.argmax(dp[T-1])

    # step2: 通過從後到前的循環來一次求出每個單詞的詞性
    for i in range(T-2,-1,-1): # T-2,T-3,...1,0
        best_seq[i]=ptr[i+1][best_seq[i+1]]

    # 到目前為止， best_seq 存放了對應x的詞性序列
    for i in range(len(best_seq)):
        print(id2tag[best_seq[i]])


x="Chicago"

print(viterbi(x,pi,A,B))
