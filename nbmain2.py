import os
import nb_helper2
import io
import json

prompt = raw_input("Enter the path for the train file\n")
classifier = nb_helper2.Classifier()

with open(prompt, 'r') as train_file:
    i=0
    for line in train_file:
        if i>3000: break
        print 'training {}th review'.format(i)
        entity = json.loads(line)
        classifier.learn(entity['text'], entity['stars'])
        i+=1
       

classifier.get_statistics()
test_path = raw_input('Enter the path for the test file\n')
output_path =  raw_input('Enter the path for the output file\n')

with open(test_path, 'r') as test_file, open(output_path, 'wb') as output_file:
    for line in test_file:
        entity = json.loads(line)
        if entity['stars'] == 3: continue
        prediction = classifier.classify(entity['text'])
        print(entity['review_id'] + " "  + str(prediction))
        outstring = {
                        'review_id': entity['review_id'],
                        'stars': prediction
                    }
                    
        json.dump(outstring, output_file)
        output_file.write('\n')
