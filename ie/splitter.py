import json
import random

poscount = 0
negcount = 0
i = 0
reviews = raw_input("Enter the path for the train file\n")
with open('all_review.json', 'r') as original, \
     open('test.json', 'wb') as test, \
     open('train.json', 'wb') as train:
    #reviews = original.readlines()
    with open(reviews, 'r') as train_file:
        for line in train_file:
            entity = json.loads(line)
            if (entity['stars'] < 3):
                if (negcount > 5000):
                    continue
                else:
                    train.write(line)                    
                    negcount += 1
                    print '{}th neg sample'.format(negcount)
                
            elif (entity['stars'] >3):
                if (poscount > 5000):
                    continue
                else:
                    train.write(line)                    
                    poscount +=1
                    print '{}th pos sample'.format(poscount)
            if(i > 10000):
                break
            else:
                i += 1
