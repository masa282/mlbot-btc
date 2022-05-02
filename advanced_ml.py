import math
import pandas as pd
import numpy as np
import multiprocessing as mp
import datetime as dt
import time
import sys

numthreads = 6

################################################################################
# Parallel (jobs)
################################################################################
def linParts(numAtoms,numThreads):
    parts=np.linspace(0, numAtoms, min(numThreads,numAtoms)+1)
    parts=np.ceil(parts).astype(int)
    return parts

def nestedParts(numAtoms,numThreads,upperTriang=False):
    parts,numThreads_=[0],min(numThreads,numAtoms)
    for num in range(numThreads_):
        part=1+4*(parts[-1]**2+parts[-1]+numAtoms*(numAtoms+1.)/numThreads_)
        part=(-1+part**.5)/2.
        parts.append(part)
    parts=np.round(parts).astype(int)
    if upperTriang:
        parts=np.cumsum(np.diff(parts)[::-1])
        parts=np.append(np.array([0]),parts)
    return parts 

def processJobs_(jobs):
    out=[]
    for job in jobs:
        out_=expandCall(job)
        out.append(out_)
    return out

def processJobs(jobs,task=None,numThreads=numthreads):
    if task is None:task=jobs[0]['func'].__name__
    pool=mp.Pool(processes=numThreads)
    outputs,out,time0=pool.imap_unordered(expandCall,jobs),[],time.time()
    for i,out_ in enumerate(outputs,1):
        out.append(out_)
        reportProgress(i,len(jobs),time0,task)
    pool.close();pool.join()
    return out

def reportProgress(jobNum,numJobs,time0,task):
    msg=[float(jobNum)/numJobs, (time.time()-time0)/60.]
    msg.append(msg[1]*(1/msg[0]-1))
    timeStamp=str(dt.datetime.fromtimestamp(time.time()))
    msg=timeStamp+' '+str(round(msg[0]*100,2))+'% '+task+' done after '+ \
        str(round(msg[1],2))+' minutes. Remaining '+str(round(msg[2],2))+' minutes.'
    if jobNum<numJobs:sys.stderr.write(msg+'\r')
    else:sys.stderr.write(msg+'\n')
    return

def expandCall(kargs):
    func=kargs['func']
    del kargs['func']
    out=func(**kargs)
    return out

def mpPandasObj(func, pdObj, numThreads=numthreads, mpBatches=1, linMols=True, **kargs):
    import pandas as pd
    if linMols:parts=linParts(len(pdObj[1]),numThreads*mpBatches)
    else:parts=nestedParts(len(pdObj[1]),numThreads*mpBatches)
    jobs=[]
    for i in range(1,len(parts)):
        job={pdObj[0]:pdObj[1][parts[i-1]:parts[i]],'func':func}
        job.update(kargs)
        jobs.append(job)
    if numThreads==1: out=processJobs_(jobs)
    else: out=processJobs(jobs, numThreads=numThreads)
    if isinstance(out[0],pd.DataFrame):df0=pd.DataFrame()
    elif isinstance(out[0],pd.Series):df0=pd.Series()
    else:return out
    for i in out: df0=pd.concat([df0, i],axis=0)
    df0=df0.sort_index()
    return df0


################################################################################
# Estimate uniquness of label
################################################################################
def mpNumCoEvents(closeIdx, t1, molecule):
    t1 = t1.fillna(closeIdx[-1])
    t1 = t1[t1>=molecule[0]]
    t1 = t1.loc[:t1[molecule].max()]
    iloc=closeIdx.searchsorted(np.array([t1.index[0],t1.max()]))
    count=pd.Series(0,index=closeIdx[iloc[0]:iloc[1]+1])
    for tIn, tOut in t1.iteritems():count.loc[tIn:tOut]+=1.
    return count.loc[molecule[0]:t1[molecule].max()]


def mpSampleTW(t1, numCoEvents, molecule):
    wght=pd.Series(index=molecule, dtype="float64")
    for tIn,tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn]=(1./numCoEvents.loc[tIn:tOut]).mean()
    return wght

def get_uniqueness(close, events, out):
    numCoEvents = mpPandasObj(func=mpNumCoEvents,
                              pdObj=("molecule", events.index),
                              numThreads=numthreads,
                              closeIdx=close.index,
                              t1=events["t1"])
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep="last")]
    out["numCoEvents"] = numCoEvents.reindex(close.index).fillna(0)
    out['tW'] = mpPandasObj(func=mpSampleTW,
                             pdObj=("molecule", events.index),
                             numThreads=numthreads,
                             t1=events["t1"],
                             numCoEvents=numCoEvents)
    return

################################################################################
# Calculate weights of samples based on estimated label's uniquness
################################################################################
def mpSampleW(t1, numCoEvents, close, molecule):
    ret=np.log(close).diff()
    wght=pd.Series(index=molecule, dtype="float64")
    for tIn,tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn]=(ret.loc[tIn:tOut]/numCoEvents.loc[tIn:tOut]).sum()
    return wght.abs()

def weighten_sample(close, events, out, numCoEvents):
    out['w'] = mpPandasObj(mpSampleW,
                           ("molecule", events.index),
                           numThreads=numthreads,
                           t1=events["t1"],
                           numCoEvents=numCoEvents,
                           close=close)
    out['w'] *= out.shape[0] / out['w'].sum()
    return


################################################################################
# Validation
################################################################################
from sklearn.model_selection._split import _BaseKFold
class PurgedKFold(_BaseKFold):
    #The train is purged of observations overlapping test-label intervals
    #Test set is assumed contiguous (shuffle=False), w/o training samples in between
    def __init__(self,n_splits=3,t1=None,pctEmbargo=0.):
        if not isinstance(t1,pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super(PurgedKFold,self).__init__(n_splits,shuffle=False,random_state=None)
        self.t1=t1
        self.pctEmbargo=pctEmbargo

    def split(self,X,y=None,groups=None):
        if (X.index==self.t1.index).sum()!=len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')
        
        indices=np.arange(X.shape[0])
        mbrg=int(X.shape[0]*self.pctEmbargo)
        
        if groups is not None: # groups are paths-list
            model_index_splits, pieces = [], []
            start_end = [(i[0],i[-1]+1) for i in np.array_split(indices, self.n_splits)] #ex.) [(0, 15066), (15066, 30131), (30131, 45196)]
            print("groups is not None!")
            for i, j in start_end:
                pieces.append(indices[i:j])                 #pieces: array-list "all of fold's indices"

            for split in range(groups.shape[1]):
                non_zero = np.nonzero(groups[:, split])
                train, test = [], []
                for piece in range(len(pieces)):            #len(pieces)==groups.shape[0]
                    if np.isin(piece, non_zero):
                        test.append(pieces[piece])
                    else:
                        train.append(pieces[piece])
                model_index_splits.append((train, test))    #model_index_splits

            #master_sets = []
            for i in range(len(model_index_splits)):        #len(model_index_splits)==nCr, ex.)15
                sets = model_index_splits[i]
                train, test = sets[0], sets[1]
                train_indices = np.concatenate(train)       
                train_t1 = self.t1.iloc[train_indices]      #indices → t1
                #adjusted_trains = []
                for i in range(len(test)):                  #len(test)==k,  ex.)2
                    test_indices = test[i].copy()           
                    #embargo
                    mbrg_indices = np.arange(test_indices[-1]+1, test_indices[-1]+mbrg)
                    test_indices = np.append(test_indices, mbrg_indices)
                    test_t1 = self.t1.iloc[np.intersect1d(indices, test_indices)]   #indices → t1
                    start, end = test_t1.index[0], test_t1[-1]                      #test_t1.index.values[0], test_t1.index.values[-1]
                    #purge
                    train_t1_0 = train_t1[(start<=train_t1.index) & (train_t1.index<=end)].index
                    train_t1_1 = train_t1[(start<=train_t1) & (train_t1<=end)].index
                    train_t1_2 = train_t1[(train_t1.index<=start) & (end<=train_t1)].index
                    train_t1 = train_t1.drop(train_t1_0.union(train_t1_1).union(train_t1_2)) #drop union of sets
                print(test)
                test_indices = np.concatenate(test)
                new_train_indices = self.t1.index.searchsorted(train_t1.index)  #convert new index to indices
                yield new_train_indices, test_indices
        else:
            print("groups is None!")
            test_starts=[(i[0],i[-1]+1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)] #ex.) [(0, 15066), (15066, 30131), (30131, 45196)]
            for i,j in test_starts:
                t0=self.t1.index[i] # start of test set
                test_indices=indices[i:j]
                maxT1Idx=self.t1.index.searchsorted(self.t1[test_indices].max())
                train_indices=self.t1.index.searchsorted(self.t1[self.t1<=t0].index)
                if maxT1Idx<X.shape[0]: # right train ( with embargo)
                    train_indices=np.concatenate((train_indices, indices[maxT1Idx+mbrg:]))
                """ like this
                [15073 15074 15075 ... 45193 45194 45195],  [    0     1     2 ... 15063 15064 15065]
                [    0     1     2 ... 45193 45194 45195],  [15066 15067 15068 ... 30128 30129 30130]
                [    0     1     2 ... 30121 30122 30123],  [30131 30132 30133 ... 45193 45194 45195]
                """
                yield train_indices,test_indices


def nCr(n, r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def cpcv_pathmap(n=6, k=2):
    col=0
    splits = int(nCr(n, n-k))
    split_map = np.zeros([n, splits])
    for base in range(n):
        for other in range(base+1, n):
            split_map[base, col] = 1
            split_map[other, col] = 1
            col += 1

    for row in range(n):
        for i in range(1, splits):
            val = split_map[row, i]
            prev_val = np.max(split_map[row, :i])
            if val == 0:
                continue
            elif val == 1:
                split_map[row, i] = prev_val+1
    return split_map


def cvScore(clf,X,y,sample_weight,scoring='neg_log_loss', t1=None,cv=None,cvGen=None, pctEmbargo=0, groups=None):
    if scoring not in ['neg_log_loss','accuracy']:
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss,accuracy_score
    if cvGen is None:
        cvGen=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged
    score=[]
    for train,test in cvGen.split(X=X, groups=groups):
        fit=clf.fit(X=X.iloc[train,:],y=y.iloc[train],sample_weight=sample_weight.iloc[train].values)
        if scoring=='neg_log_loss':
            prob=fit.predict_proba(X.iloc[test,:])
            score_=-log_loss(y.iloc[test], prob, sample_weight=sample_weight.iloc[test].values, labels=clf.classes_)
        else:
            pred=fit.predict(X.iloc[test,:])
            score_=accuracy_score(y.iloc[test],pred, sample_weight=sample_weight.iloc[test].values)
        score.append(score_)
    return np.array(score)


from sklearn.pipeline import Pipeline
class MyPipeline(Pipeline):
    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0]+"__sample_weight"] = sample_weight
        return super(MyPipeline, self).fit(X, y, **fit_params)



################################################################################
# Corwin-Schultz Algorithm
################################################################################
def getBeta(series, s1):
    h1 = series[["high", "low"]].values
    h1 = np.log(h1[:,0]/h1[:,1])**2
    h1 = pd.Series(h1, index=series.index)
    #beta = pd.stats.moments.rolling_sum(h1, window=2)
    #beta = pd.stats.moments.rolling_mean(beta, window=s1)
    beta = h1.rolling(window=2).sum()
    beta = beta.rolling(window=s1).mean()
    return beta.dropna()

def getGamma(series):
    #h2 = pd.stats.moments.rolling_max(series["high"], window=2)
    #l2 = pd.stats.moments.rolling_min(series["low"], window=2)
    h2 = series["high"].rolling(window=2).mean()
    l2 = series["low"].rolling(window=2).min()
    gamma = np.log(h2.values/l2.values) ** 2
    gamma = pd.Series(gamma, index=h2.index)
    return gamma.dropna()

def getAlpha(beta, gamma):
    den = 3 - 2*2**.5
    alpha = (2**.5-1)*(beta**.5)/den
    alpha -= (gamma/den)**.5
    alpha[alpha<0] = 0
    return alpha.dropna()

def corwinSchultz(series, s1=1):
    beta = getBeta(series, s1)
    gamma = getGamma(series)
    alpha = getAlpha(beta, gamma)
    spread = 2*(np.exp(alpha)-1) / (1+np.exp(alpha))
    startTime = pd.Series(series.index[0:spread.shape[0]], index=spread.index)
    spread = pd.concat([spread, startTime], axis=1)
    spread = pd.concat([spread, beta], axis=1)
    spread = pd.concat([spread, gamma], axis=1)
    spread.columns = ["Spread", "Start_Time", "beta", "gamma"]
    return spread
