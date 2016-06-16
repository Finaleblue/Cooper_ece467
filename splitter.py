#Cooper Union Natural Language Processing Final Project
#Author: Eui Han
#Version: 04/26/16

import json
import random

history = set() 
poscount = 0
negcount = 0
with open('all_review.json', 'r') as original, \
     open('test.json', 'wb') as test, \
     open('train.json', 'wb') as train:
    reviews = original.readlines()
    while poscount < 1500 or negcount < 1500:
        index = random.randint(0,2225212)
        if index in history:
            continue
        entity = json.loads(reviews[index])
        if (entity['stars'] < 3):
            if (negcount > 1500):
                continue
            else:
                train.write(reviews[index])
                history.add(index)
                negcount += 1
                print '{}th neg sample'.format(negcount)
            
        elif (entity['stars'] >3):
            if (poscount > 1500):
                continue
            else:
                train.write(reviews[index])
                history.add(index)
                poscount +=1
                print '{}th pos sample'.format(poscount)

    poscount = 0
    negcount = 0
    while poscount < 500 or negcount < 500:
        index = random.randint(0,2225212)
        if index in history:
            continue
        entity = json.loads(reviews[index])
        if entity['stars'] < 3 :
            if negcount > 500 : continue
            test.write(reviews[index])
            history.add(index)
            negcount += 1
            print '{}th neg testcase'.format(negcount)
        elif entity['stars'] > 3 :
            if poscount > 500 : continue
            test.write(reviews[index])
            history.add(index)
            poscount += 1
            print '{}th pos testcase'.format(poscount)
