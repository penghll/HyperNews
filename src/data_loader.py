import numpy as np
import pandas as pd
import json
import math

from collections import namedtuple

Data = namedtuple('Data',['size','news_title','clicked_titles','active_time','labels',
                          'news_category','user_city','user_region','clicked_category',
                          'time_label','news_age','news_id','news_len','clicked_news_id',
                          'clicked_news_len','is_train','weights'])

# time_divide=[0,5,13,22,33,45,60,79,108,165,900]
# time_divide=[5,12,20,30,40,52,67,87,117,178,900]
time_divide=[5,15,25,35,45,55,65,75,85,95,105,115,125,135,145,155,165,175,185,195,205]

def get_age_id(age):
    if age <= 0:
        age_id=24+30+12+20+1
    elif age < 3600*24:
        age_id=age/3600
    elif age < 3600*24*30:
        age_id=age/(3600*24)+24
    elif age < 3600*24*30*12:
        age_id=age/(3600*24*30)+24+30
    elif age < 3600*24*30*12*20:
        age_id=age/(3600*24*30*12)+24+30+12
    else:
        age_id=24+30+12+20
    return age_id

def load_data(args, history_list):

    import time
    time_stamp = int(time.time())

    user_id=[]
    news_id=[]
    time=[]
    active_time=[]
    labels=[]
    
    #time_nums_dic=json.loads(open('../data/content_data/time_nums','r',encoding='utf-8').read())
    #time_nums=[0]*20
    #for i in range(5,206):
    #    for j in range(len(time_divide)-1):
    #        if i < time_divide[j+1]:
    #            time_nums[j] = time_nums[j]+time_nums_dic[str(i)]
    #            break
    #print(time_nums)

    f1=open('../data/content_data/newsid2titleid','r',encoding='utf-8')
    f2=open('../data/content_data/newsid2categoryid','r',encoding='utf-8')
    f3=open('../data/content_data/newsid2publishtime','r',encoding='utf-8')
    f4=open('../data/content_data/newsid2len','r',encoding='utf-8')
    news_id_to_title=json.loads(f1.read())
    news_id_to_category=json.loads(f2.read())
    news_id_to_publishtime=json.loads(f3.read())
    news_id_to_len=json.loads(f4.read())
    f1.close()
    f2.close()
    f3.close()
    f4.close()
	
    f1 = open('../data/user_infos/id2history','r',encoding='utf-8')
    f2 = open('../data/user_infos/id2city', 'r', encoding='utf-8')
    f3 = open('../data/user_infos/id2region', 'r', encoding='utf-8')
    user_id_to_history=json.loads(f1.read())
    user_id_to_city=json.loads(f2.read())
    user_id_to_region=json.loads(f3.read())
    f1.close()
    f2.close()
    f3.close()
	
    # get base event data
    for id in range(1, 74886):

        user_id.append(str(0))
        news_id.append(id)
        time.append(time_stamp)
        active_time.append(0)
        labels.append(0)
    #news_title=[]


    # get news title, category..
    news_title=np.zeros(dtype=np.int,shape=(len(labels),args.max_title_length))
    news_category=np.zeros(dtype=np.int,shape=(len(labels),3))
    news_age=np.zeros(dtype=np.int,shape=(len(labels),1))
    news_len=np.zeros(dtype=np.int,shape=(len(labels),1))

    index=0
    for id in news_id:
        # title
        title=news_id_to_title[id]
        if len(title) > args.max_title_length:
            news_title[index]=np.array(title[0:args.max_title_length])
        else:
            # print(news_id_to_title[id])
            # news_title[index]=np.array(title.extend(0 for _ in range(args.max_title_length-len(news_id_to_title[id]))))
            # news_title[index] = np.array(news_id_to_title[id])
            np_title=np.zeros(dtype=np.int,shape=(args.max_title_length,))
            for i in range(len(title)):
                np_title[i]=title[i]
            news_title[index]=np_title
        # print(news_title[index])

        # category
        category=news_id_to_category[id]
        for i in range(len(category)):
            if i>=3:
                break
            news_category[index][i]=category[i]

        # age
        age=time[index]-news_id_to_publishtime[id]
        news_age[index][0]=get_age_id(age)

        # length
        news_len[index][0]=news_id_to_len[id]/50+1

        # index
        index=index+1

    # get user's click history, city, region..
    clicked_titles=np.zeros(dtype=np.int,shape=(len(labels),args.max_click_history,args.max_title_length))
    clicked_category=np.zeros(dtype=np.int,shape=(len(labels),args.max_click_history,3))
    clicked_news_id=np.zeros(dtype=np.int,shape=(len(labels),args.max_click_history,1))
    clicked_news_len=np.zeros(dtype=np.int,shape=(len(labels),args.max_click_history,1))
    user_city=np.zeros(dtype=np.int,shape=(len(labels),1))
    user_region=np.zeros(dtype=np.int,shape=(len(labels),1))

    index=0
    for id in user_id:
        # clicked title&category&id&len
        titles=np.zeros(dtype=np.int,shape=(args.max_click_history,args.max_title_length))
        # news=np.zeros(dtype=np.int,shape=(args.max_click_history,1))
        # len=np.zeros(dtype=np.int,shape=(args.max_click_history,1))
        i=0
        history=history_list
        if len(history) > args.max_click_history:
            history=history[-1*args.max_click_history:]
        for h in history:
            # title
            title=news_id_to_title[str(h)]
            if len(title) > args.max_title_length:
                titles[i]=np.array(title[0:args.max_title_length])
            else:
                np_title=np.zeros(dtype=np.int,shape=(args.max_title_length,))
                for j in range(len(title)):
                    np_title[j]=title[j]
                titles[i]=np_title
            # category
            category=news_id_to_category[str(h)]
            for j in range(len(category)):
                if j>=3:
                    break
                clicked_category[index][i][j]=category[j]
            # news id
            clicked_news_id[index][i]=h
            # news len
            clicked_news_len[index][i]=news_id_to_len[str(h)]/50+1

            i=i+1
        clicked_titles[index]=titles

        # city
        user_city[index][0]=user_id_to_city[id]

        # region
        user_region[index][0]=user_id_to_region[id]

        # index
        index=index+1

    # set time_label
    
    time_nums_dic=json.loads(open('../data/content_data/time_nums','r',encoding='utf-8').read())
    time_nums=[0]*20
    for i in range(5,206):
        for j in range(len(time_divide)-1):
            if i < time_divide[j+1]:
                time_nums[j] = time_nums[j]+time_nums_dic[str(i)]
                break
    #print(time_nums)
    beta=0.99999


    weights=[0]*20
    for i in range(20):
        weights[i]=(1-beta)/(1-math.pow(beta,time_nums[i]))

    weights_sum=sum(weights)
    for i in range(20):
        weights[i]=weights[i]/weights_sum

    #print(weights)

    weighted_val=np.zeros(dtype=np.float,shape=(len(active_time),1))

    active_time_input=np.zeros(dtype=np.int,shape=(len(active_time),20))
    time_label=np.zeros(dtype=np.float,shape=(len(active_time),1))
    for i in range(len(active_time)):
        time=active_time[i]
        if time >= 5 and time <=205:
            for j in range(len(time_divide)-1):
                if time < time_divide[j+1]:
                    active_time_input[i][j]=1
                    weighted_val[i][0]=weights[j]
                    break
        else:
            time_label[i][0]=1

    data=Data(size=len(labels),
              news_title=news_title,
              clicked_titles=clicked_titles,
              active_time=active_time_input,
              labels=np.array(labels).reshape(-1,1),
              time_label=time_label,
              news_category=news_category,
              user_city=user_city,
              user_region=user_region,
              clicked_category=clicked_category,
              news_age=news_age,
              news_id=np.array(news_id).reshape(-1,1),
              news_len=news_len,
              clicked_news_id=clicked_news_id,
              clicked_news_len=clicked_news_len,
              is_train=args.is_train,
              weights=weighted_val)

    return data
