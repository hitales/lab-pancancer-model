import shap
shap.initjs()
import matplotlib.pyplot as plt
import math
import pickle
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_curve,auc
from xgboost.sklearn import XGBClassifier
import sklearn.metrics as metrics
import numpy as np
import os
target_tag = '肠癌'

data_input = "C:/tests/faz20220609/v2/pyout/datas/"
model_input = "C:/tests/faz20220609/v2/pyout/models_scale/"
save_path = 'C:/tests/faz20220609/v2/pyout/reports/'+target_tag+"/"
if (not os.path.exists(save_path)):
    os.makedirs(save_path)
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.figure(figsize=(10,10))

file = open(model_input+target_tag+".pkl", 'rb+')
model_infos = pickle.load(file)
feature_names = model_infos['feature_names']
model = model_infos['model']
file.close()

file = open(data_input+target_tag+".pkl", 'rb+')
data_infos = pickle.load(file)
file.close()

def filter(datas,selects):
    datas2 = []
    for i in range(0,len(datas)):
        data = []
        for j in range(0,len(selects)):
            data.append(datas[i][selects[j]])
        datas2.append(data)
    return np.array(datas2)

datas_train = filter(data_infos['train_datas'],model_infos['selects'])
tags_train = data_infos['train_tags']
datas_valid = filter(data_infos['valid_datas'],model_infos['selects'])
tags_valid = data_infos['valid_tags']
datas_test = filter(data_infos['test_datas'],model_infos['selects'])
tags_test = data_infos['test_tags']

print(len(datas_train),len(datas_train[0]))
print(model.predict(datas_train))
d2 = {}
for i in range(0,len(feature_names)):
    d2[feature_names[i]] = []
for i in range(0,len(datas_train)):
    for j in range(0,len(feature_names)):
        d2[feature_names[j]].append(datas_train[i][j])

df1 = pd.DataFrame(d2)
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(df1.values)

fig = plt.figure(figsize=(12, 6), dpi=150)
shap.summary_plot(shap_values, df1, show=False,max_display=30)
fig.tight_layout()
plt.savefig(save_path+'特征权重图.png', dpi=150)
plt.clf()

fig = plt.figure(figsize=(30, 6), dpi=150)
shap.summary_plot(shap_values, df1, plot_type="bar", show=False,max_display=30)
fig.tight_layout()
plt.savefig(save_path+'特征权重图2.png', dpi=150)
plt.clf()
plt.close()

def get_coefs(model,mtype):
    if(mtype=='lr'):
        return model.coef_[0]
    if(mtype=='xgb'):
        importance = model.get_booster().get_score(importance_type='gain')
        res = []
        for i in range(0,len(feature_names)):
            res.append(0)
        for key in importance:
            idx = int(key[1:])
            res[idx] = importance[key]
        return res
    return None

import xlwt
vs = get_coefs(model,'xgb')
workbook = xlwt.Workbook()
sheet = workbook.add_sheet("模型参数")
sheet.write(0, 0, '特征名称')
sheet.write(0, 1, '特征权重')

for i in range(0,len(vs)):
    sheet.write(i+1,0,feature_names[i])
    sheet.write(i+1,1,vs[i])
workbook.save(save_path+'模型特征权重值.xls')

workbook0 = xlwt.Workbook()
sheet0 = workbook0.add_sheet("模型参数shap权重")
sheet0.write(0, 0, '参数名称')
sheet0.write(0, 1, '权重')
arr2 = []
for j in range(0, len(feature_names)):
    vl = 0
    for k in range(0, len(shap_values)):
        vl = vl + abs(shap_values[k][j])
    vl = vl / len(shap_values)
    # print(feature_names[j],vl,float(vl))
    arr2.append({'name': feature_names[j], "value": float(vl)})

arr2.sort(key=lambda x: x['value'], reverse=True)
for j in range(0, len(arr2)):
    sheet0.write(j + 1, 0, arr2[j]['name'])
    sheet0.write(j + 1, 1, arr2[j]['value'])
workbook0.save(save_path + 'shap特征权重值.xls')

def get_threshold(rec,tags,prs):
    prs2 = []
    for i in range(0,len(tags)):
        if(tags[i]==1):
            prs2.append(prs[i])
    prs2.sort()
    # print(prs2)
    return prs2[math.floor(len(prs2)*(1-rec))]

def get_zhibiao2(tags,prs,thresh):
    tf = 0
    ff = 0
    tt = 0
    ft = 0
    np = 0
    for i in range(0, len(tags)):
        if (tags[i] == 0):
            if (prs[i] >=thresh):
                ft = ft+1
            else:
                ff = ff+1
        else:
            np = np +1
            if (prs[i] >= thresh):
                tt = tt + 1
            else:
                tf = tf + 1
    fpr, tpr, thresholds = roc_curve(tags,prs)
    ascore = auc(fpr, tpr)
    prec = 0
    if(ft+tt>0):
        prec = 1.0*tt/(ft+tt)
    rec = 0
    if(tf+tt>0):
        rec = 1.0*tt/(tf+tt)
    ty = 0
    if(ff+ft>0):
        ty = 1.0*ff/(ff+ft)
    acc = 0
    acc = (ff+tt)*1.0/(ff+tt+tf+ft)
    return {'敏感性':round(rec,4),'auc':round(ascore,4),'特异性':ty,"精准率":round(prec,4),"样本数":len(tags),
            "真阴性":ff,"假阴性":tf,"真阳性":tt,"假阳性":ft,"阳性样本数":np,'准确率':round(acc,4)}

plt.clf()
pr1 = model.predict_proba(datas_train)[:,1]
fpr, tpr, thresholds = roc_curve(tags_train, pr1)
ascore = auc(fpr, tpr)
plt.figure(figsize=(8, 8))
plt.title('ROC')
plt.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % ascore)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(save_path+'训练集_AUC.png')
plt.close()

plt.clf()
pr2 = model.predict_proba(datas_valid)[:,1]
fpr, tpr, thresholds = roc_curve(tags_valid, pr2)
ascore = auc(fpr, tpr)
plt.figure(figsize=(8, 8))
plt.title('ROC')
plt.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % ascore)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(save_path+'验证集_AUC.png')
plt.close()

plt.clf()
pr3 = model.predict_proba(datas_test)[:,1]
fpr, tpr, thresholds = roc_curve(tags_test, pr3)
ascore = auc(fpr, tpr)
plt.figure(figsize=(8, 8))
plt.title('ROC')
plt.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % ascore)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(save_path+'测试集_AUC.png')
plt.close()

workbook2 = xlwt.Workbook()
sheet2 = workbook2.add_sheet("模型效果")
sheet2.write(0, 0, '数据集')

headers = ['阈值','样本数','阳性样本数','auc','敏感性','特异性','精准率','准确率','真阴性','假阴性','假阳性','真阳性']

for i in range(0,len(headers)):
    sheet2.write(0,i+1,headers[i])

def get_yd_tr(tags,prs):
    fpr, tpr, thresholds = roc_curve(tags, prs)
    maxv = -1
    mtr = 0.5
    for i in range(0,len(fpr)):
        v = tpr[i]-fpr[i]
        if(v>maxv):
            maxv = v
            mtr = thresholds[i]
        # print(i,fpr[i],tpr[i],v)
    return maxv,mtr

def write_sheet(starti,key,datas,tags):
    sheet2.write(starti,0,key)
    sheet2.write(starti+1,0,key+'(0.85敏感)')
    sheet2.write(starti + 2, 0, key + '(0.75敏感)')
    sheet2.write(starti+3,0,key+'(约登指数最大)')
    pr0 = model.predict_proba(datas)[:, 1]
    rp1 = get_zhibiao2(tags, pr0, 0.5)
    for i in range(1,len(headers)):
        sheet2.write(starti,i+1,rp1[headers[i]])
    sheet2.write(starti, 1, str(0.5))

    tr2 = get_threshold(0.85, tags, pr0)
    rp2 = get_zhibiao2(tags, pr0, tr2)
    for i in range(1,len(headers)):
        sheet2.write(starti+1,i+1,rp2[headers[i]])
    sheet2.write(starti+1, 1, str(round(tr2,4)))

    tr3 = get_threshold(0.75, tags, pr0)
    rp3 = get_zhibiao2(tags, pr0, tr3)
    for i in range(1, len(headers)):
        sheet2.write(starti+2, i + 1, rp3[headers[i]])
    sheet2.write(starti + 2, 1, str(round(tr3, 4)))

    maxv,tr4 = get_yd_tr(tags,pr0)
    rp4 = get_zhibiao2(tags, pr0, tr4)
    for i in range(1, len(headers)):
        sheet2.write(starti + 3, i + 1, rp4[headers[i]])
    sheet2.write(starti + 3, 1, str(round(tr4, 4))+",此时约登指数为"+str(round(maxv,4)))

write_sheet(1,'训练集',datas_train,tags_train)
write_sheet(5,'验证集',datas_valid,tags_valid)
write_sheet(9,'测试集',datas_test,tags_test)
#
# sheet2.write(1,0,'训练集')
# sheet2.write(2,0,'训练集(0.85敏感)')
# sheet2.write(3,0,'验证集')
# sheet2.write(4,0,'验证集(0.85敏感)')
# sheet2.write(5,0,'测试集')
# sheet2.write(6,0,'测试集(0.85敏感)')

# rp1 = get_zhibiao2(tags_train,pr1,0.5)
# for i in range(1,len(headers)):
#     sheet2.write(1,i+1,rp1[headers[i]])
# tr1 = get_threshold(0.85,tags_train,pr1)
# rp2 = get_zhibiao2(tags_train,pr1,tr1)
# for i in range(1,len(headers)):
#     sheet2.write(2,i+1,rp2[headers[i]])
#
# rpt1 = get_zhibiao2(tags_valid,pr2,0.5)
# for i in range(1,len(headers)):
#     sheet2.write(3,i+1,rpt1[headers[i]])
#
# tr2 = get_threshold(0.85,tags_valid,pr2)
# rpt2 = get_zhibiao2(tags_valid,pr2,tr2)
# for i in range(1,len(headers)):
#     sheet2.write(4,i+1,rpt2[headers[i]])
#
# rpt_yb = get_zhibiao2(tags_test,pr3,0.5)
# for i in range(1,len(headers)):
#     sheet2.write(5,i+1,rpt_yb[headers[i]])
#
# tr3 = get_threshold(0.85,tags_test,pr3)
# rpt_yb2 = get_zhibiao2(tags_test,pr3,tr3)
# for i in range(1,len(headers)):
#     sheet2.write(6,i+1,rpt_yb2[headers[i]])
#
# sheet2.write(1,1,0.5)
# sheet2.write(2,1,float(round(tr1,4)))
# sheet2.write(3,1,0.5)
# sheet2.write(4,1,float(round(tr2,4)))
# sheet2.write(5,1,0.5)
# sheet2.write(6,1,float(round(tr2,tr3)))
workbook2.save(save_path+'模型效果.xls')

print('---DONE---')