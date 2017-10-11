import pandas as pd
import math

def get_entropy(data):
    entropy = 0
    total = sum(data)
    for i in data:
        entropy -= i/total * math.log(i/total, 2)
    return round(entropy, 4)

simpson_data = pd.read_csv('./data/simpson.csv',
                           names=["person", "hair_length", "Weight", "age", "class"], encoding='utf-8')
print(simpson_data)
# print(get_entropy([4,5]))
# Use Hair_length

k = 5   #기준점
t_hair = simpson_data[simpson_data['hair_length']<= k]   #k보다 작은 행추출
f_hair = simpson_data.drop(t_hair.index)  #s이외의 값

# 선별된 데이터 분류
count_tM = len(t_hair[t_hair['class']=='M'])
count_tF = len(t_hair[t_hair['class']=='F'])
t_entropy = get_entropy([count_tM,count_tF])

count_fM = len(f_hair[f_hair['class']=='M'])
count_fF = len(f_hair[f_hair['class']=='F'])
f_entropy = get_entropy([count_fM,count_fF])

# 전체
total = len(simpson_data)
# print("total",total)
M = simpson_data[simpson_data['class']=='M']
count_M = len(M)
F = simpson_data[simpson_data['class']=='F']
count_F = len(F)

entropy = get_entropy([count_M,count_F])

gain = 0
gain = entropy - (((count_tM+count_tF)/total)*t_entropy + ((count_fM+count_fF)/total)*f_entropy)
print(round(gain,4))
