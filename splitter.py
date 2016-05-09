import json
import random

history = set() 
poscount = 0
negcount = 0
with open('all_review.json', 'r') as original, \
     open('test.json', 'wb') as test, \
     open('train.json', 'wb') as train:
    reviews = original.readlines()
    while poscount < 5000 or negcount < 5000:
        index = random.randint(0,2225212)
        if index in history:
            continue
        entity = json.loads(reviews[index])
        if (entity['stars'] < 3):
            if (negcount > 5000):
                continue
            else:
                train.write(reviews[index])
                history.add(index)
                negcount += 1
                print '{}th neg sample'.format(negcount)
            
        elif (entity['stars'] >3):
            if (poscount > 5000):
                continue
            else:
                train.write(reviews[index])
                history.add(index)
                poscount +=1
                print '{}th pos sample'.format(poscount)

    for i in range(1000):
        index = random.randint(0,2225212)
        if index in history:
            continue
        else:
            test.write(reviews[index])
