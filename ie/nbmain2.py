import os
import nb_helper2
import io
import json
import nltk

#prompt = raw_input("Enter the path for the train file\n")
classifier = nb_helper2.Classifier()
prompt = 'train.json'
with open(prompt, 'r') as train_file:
    i=0
    for line in train_file:
        if i>1000: break
        print 'training {}th review'.format(i)
        entity = json.loads(line)

           #print entity['text']
        classifier.learn(entity['text'], entity['stars'])
        #    break


        i+=1

classifier.get_statistics()
test_path = raw_input('Enter the path for the test file\n')
output_path =  raw_input('Enter the path for the output file\n')

with open(test_path, 'r') as test_file, open(output_path, 'wb') as output_file:
    j = 0
    for line in test_file:
        if j>400: break
        entity = json.loads(line)
        if entity['stars'] == 3: continue
        prediction = classifier.classify(entity['text'])
        print 'predicting {}th review'.format(j)
        print(entity['review_id'] + " "  + str(prediction))
        outstring = {
                        'review_id': entity['review_id'],
                        'stars': prediction
                    }
                    
        json.dump(outstring, output_file)
        output_file.write('\n')
        j+=1
