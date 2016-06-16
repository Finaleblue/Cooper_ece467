#Cooper Union Natural Language Processing Final Project
#Author: Eui Han
#Version: 04/26/16

from __future__ import division
import sys
import json

arg1 = sys.argv[1]
arg2 = sys.argv[2]
success = 0
total = 0
poscount = 0
negcount = 0
with open(arg1, 'r') as output:
    for line in output:
        entity1 = json.loads(line)
        with open(arg2, 'r') as reference:
            for target in reference:
                entity2 = json.loads(target)
                if (entity1['review_id'] == entity2['review_id']):
                    print('review id: {}, stars: {}, predict: {}' \
                            .format(entity1['review_id'], entity2['stars'], entity1['stars']))
                    if entity2['stars'] > 3 : poscount += 1
                    elif entity2['stars'] < 3 : negcount += 1
                    if (entity1['stars'] == 'pos' \
                            and entity2['stars']>3):
                        success += 1
                    elif(entity1['stars'] == 'neg' \
                            and entity2['stars']<3):
                        success += 1
        total += 1

print('success rate: {} \ntotal:{}'.format(success/total, total))
print('pos: {}, neg: {}'.format(poscount, negcount))
