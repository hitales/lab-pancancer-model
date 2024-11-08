import random
import numpy as np
from xgboost.sklearn import XGBClassifier
import pickle
import sys
import time
import warnings
warnings.filterwarnings("ignore")
import random
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_curve,auc

target_tag = '肠癌'

input = "C:/tests/faz20220609/v2/pyout/datas/"
output = "C:/tests/faz20220609/v2/pyout/models_scale/"

file = open(input+target_tag+".pkl",'rb+')
datas = pickle.load(file)
file.close()
print(datas.keys())

# change models
model = XGBClassifier(n_estimators=30,max_depth=2, scale_pos_weight=20,use_label_encoder=True,eval_metric='auc')

model.fit(datas['train_datas'],datas['train_tags'])

def get_coefs(model,mtype):
    if(mtype=='lr'):
        return model.coef_[0]
    if(mtype=='xgb'):
        importance = model.get_booster().get_score(importance_type='gain')
        res = []
        for i in range(0,len(datas['feature_names'])):
            res.append(0)
        for key in importance:
            idx = int(key[1:])
            res[idx] = importance[key]
        return res
    return None

scores0 = get_coefs(model,'xgb')
scores = []
for i in range(0,len(datas['feature_names'])):
    scores.append([i,datas['feature_names'][i],float(scores0[i])])
scores.sort(key=lambda x: x[2],reverse=True)
print(scores)
select = []
select_names = []



def filter(datas,selects):
    datas2 = []
    for i in range(0,len(datas)):
        data = []
        for j in range(0,len(selects)):
            data.append(datas[i][selects[j]])
        datas2.append(data)
    return np.array(datas2)

def get_zhibiao2(model,test_data,test_tag):
    p1 = model.predict(test_data)
    p2 = model.predict_proba(test_data)
    p_yc = precision_score(test_tag, p1)
    r_yc = recall_score(test_tag, p1)
    fpr, tpr, thresholds = roc_curve(test_tag, p2[:, 1])
    ascore = auc(fpr, tpr)
    c1 = 0
    c2 = 0
    for i in range(0,len(test_tag)):
        if(test_tag[i]==0):
            c1 = c1+1
            if(p1[i]==0):
                c2 = c2+1
    tf = 0
    ff = 0
    tt = 0
    ft = 0
    pcount = 0
    for i in range(0, len(test_tag)):
        if (test_tag[i] == 0):
            c1 = c1 + 1
            if (p1[i] == 0):
                c2 = c2 + 1
            if (p1[i] == 0):
                tf = tf + 1
            else:
                ft = ft + 1
        else:
            pcount = pcount+1
            if (p1[i] == 0):
                ff = ff + 1
            else:
                tt = tt + 1
    ty = 0
    if(c1>0):
        ty = round(c2/c1,4)
    return {'敏感性':round(r_yc,4),'auc':float(ascore),'特异性':ty,"精准率":round(p_yc,4),"样本数":len(test_tag),
            "真阴性":tf,"假阴性":ff,"真阳性":tt,"假阳性":ft,"阳性样本数":pcount}


auc_list = []

model_datas = {}
for i in range(0,len(scores)):
    select.append(scores[i][0])
    select_names.append(scores[i][1])

    datast = filter(datas['train_datas'],select)
    datasv = filter(datas['valid_datas'],select)
    model = XGBClassifier(n_estimators=30, max_depth=2, use_label_encoder=True,scale_pos_weight=20,eval_metric='auc')
    # model = LogisticRegression(penalty='l1',solver='liblinear')
    # model = SVC(kernel='linear', probability=True, max_iter=100)
    model.fit(datast, datas['train_tags'])
    report = get_zhibiao2(model, datasv, datas['valid_tags'])
    auc0 = report['auc']
    if(len(select)>10):
        valid = False
        for j in range(0,3):
            idx = len(auc_list)-1-j
            if(idx<0):
                break
            if(auc0>=auc_list[idx]):
                valid = True
        if(not valid):
            select.pop()
            select_names.pop()
            break

    auc_list.append(auc0)
    model_datas['model'] = model
    model_datas['selects'] = select
    model_datas['feature_names'] = select_names
    model_datas['auc'] = auc0
    print(len(select_names),select_names,auc0)
    if (len(select) >= 25):
        break

file = open(output+target_tag+".pkl", 'wb+')
pickle.dump(model_datas,file)
file.close()
print('---DONE---')