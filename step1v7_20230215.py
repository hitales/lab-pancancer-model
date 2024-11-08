import random
import numpy as np
from xgboost.sklearn import XGBClassifier
import pickle
import sys
import time

input = "C:/tests/faz20220609/out/"
output = "C:/tests/faz20220609/pyout/datas2/"

path_train = "C:/tests/faz20220609/out/datas_train.csv"
path_valid = "C:/tests/faz20220609/out/datas_valid.csv"
path_test = "C:/tests/faz20220609/out/datas_test.csv"

datas_train = {}
datas_valid = {}
datas_test = {}

target_tag = '乳腺癌'

def readData(path,datas):
    fi = open(path, 'r+', encoding='utf-8')
    i0 = 0
    data_datas = []
    data_tags = []
    test_indexes = {}
    inverted_test_indexes = {}
    while 1:
        lines = fi.readlines(30000)
        if not lines:
            break
        for line in lines:
            i0 = i0 + 1
            # if (i0 % 10000 == 0):
            #     print('Processing Datas', i0)
            line = line.replace("\n", "")
            arr = line.split(",")
            if (i0 > 1):
                key = arr[0] + "_" + arr[1]
                ctag = 0
                tags = set()
                a = arr[2].split(";")
                for tag in a:
                    tags.add(tag)
                if(target_tag in tags):
                    ctag = 1

                ignores = set()
                a = arr[3].split(";")
                for tag in a:
                    idx = tag.find('=')
                    tag0 = tag[0:idx]
                    ignores.add(tag0)

                # if(target_tag in ignores):
                #     print("here11111")
                # if(arr[3].__contains__("肠癌")):
                #     print(arr[3],ignores)
                if(ctag==0 and target_tag in ignores):
                    continue

                if(float(arr[test_indexes['性别']+4])==1):
                    continue

                if (float(arr[test_indexes['年龄'] + 4]) <= 14):
                    continue

                data = {}
                for j in range(4,len(arr)):
                    if (len(arr[j]) > 0):
                        v = float(arr[j])
                        data[j - 4] = v
                data_datas.append(data)
                data_tags.append(ctag)
            else:
                for j in range(4, len(arr)):
                    test_indexes[arr[j]] = len(test_indexes)
                    inverted_test_indexes[len(inverted_test_indexes)] = arr[j]
    datas['datas'] = data_datas
    datas['tags'] = data_tags
    datas['test_indexes'] = test_indexes
    datas['inverted_test_indexes'] = inverted_test_indexes


readData(path_train,datas_train)
print(len(datas_train['datas']),datas_train['tags'].count(1))
test_counter = {}
pos_counter = {}
valid_test = []

for t in datas_train['test_indexes']:
    test_counter[t] = 0
    pos_counter[t] = 0

pos_count = 0
size0 = 0
for i in range(0, len(datas_train['datas'])):
    pos = 0
    size0 = size0 + 1
    if(datas_train['tags'][i]==1):
        pos_count = pos_count+1
        pos = 1
    for t in datas_train['datas'][i]:
        test_counter[datas_train['inverted_test_indexes'][t]] = test_counter[datas_train['inverted_test_indexes'][t]] + 1
        if(pos==1):
            pos_counter[datas_train['inverted_test_indexes'][t]] = pos_counter[datas_train['inverted_test_indexes'][t]] + 1

for t in pos_counter:
    v = pos_counter[t]*1.0/pos_count
    v1 = test_counter[t]*1.0/size0
    if(v==0 or v1==0):
        continue
    v2 = v/v1
    if(v>=0.5):
        valid_test.append([t,float(v),float(v2)])

valid_test.sort(key=lambda x: x[1],reverse=True)
rz_test = set()
for i in range(0,20):
    if(i>=len(valid_test)):
        break
    rz_test.add(valid_test[i][0])
valid_test.sort(key=lambda x: x[2],reverse=True)
for i in range(0,20):
    if(i>=len(valid_test)):
        break
    rz_test.add(valid_test[i][0])

import pandas as pd
valid_test_scores = []
pgc = 0
for d in valid_test:
    pgc = pgc+1
    if(pgc%10==0):
        print('Processing score calc',pgc,'/',len(valid_test))
    idx = datas_train['test_indexes'][d[0]]
    arr1 = []
    arr2 = []
    for i in range(0,len(datas_train['datas'])):
        if(datas_train['datas'][i].__contains__(idx)):
            arr1.append(float(datas_train['datas'][i][idx]))
            arr2.append(float(datas_train['tags'][i]))
    p1 = pd.Series(arr1)
    p2 = pd.Series(arr2)
    s0 = p1.corr(p2, method='pearson')
    s = abs(s0)
    if (not np.isnan(s)):
        valid_test_scores.append({'name': d[0], 'score': s})
valid_test_scores.sort(key=lambda x: x['score'],reverse=True)
for i in range(0,120):
    if(i>=len(valid_test_scores)):
        break
    rz_test.add(valid_test_scores[i]['name'])
print(pos_count, len(datas_train['datas']))


select_features = []
select_feature_names = []

for i in range(0,len(datas_train['test_indexes'])):
    t = datas_train['inverted_test_indexes'][i]
    if(t in rz_test):
        select_features.append(i)
        select_feature_names.append(t)

print('最终入组',select_features,select_feature_names)

final_datas = {}
final_datas['feature_names'] = select_feature_names

def filter_data(datas):
    ndatas = []
    for i in range(0,len(datas)):
        ndata = []
        for j in range(0,len(select_features)):
            if(datas[i].__contains__(select_features[j])):
                ndata.append(select_features[j])
            else:
                if (select_feature_names[j] == '血_游离/总前列腺特异性抗原_V'):
                    ndata.append(random.uniform(0.25, 1.0))
                else:
                    if (select_feature_names[j].endswith('_V')):
                        ndata.append(random.uniform(0.0, 1.0))
                    else:
                        ndata.append(0.0)
        ndatas.append(ndata)
    return np.array(ndatas)

final_datas['train_datas'] = filter_data(datas_train['datas'])
final_datas['train_tags'] = np.array(datas_train['tags'])
print(len(select_features),len(final_datas['train_datas']),len(final_datas['train_datas'][0]),len(final_datas['train_tags']))
readData(path_valid,datas_valid)
final_datas['valid_datas'] = filter_data(datas_valid['datas'])
final_datas['valid_tags'] = np.array(datas_valid['tags'])
print(len(select_features),len(final_datas['valid_datas']),len(final_datas['valid_datas'][0]),len(final_datas['valid_tags']))
readData(path_test,datas_test)
final_datas['test_datas'] = filter_data(datas_test['datas'])
final_datas['test_tags'] = np.array(datas_test['tags'])
print(len(select_features),len(final_datas['test_datas']),len(final_datas['test_datas'][0]),len(final_datas['test_tags']))
file = open(output+target_tag+".pkl", 'wb+')
pickle.dump(final_datas,file)
file.close()
print('---DONE---')