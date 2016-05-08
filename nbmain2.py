import os
import nb_helper2
import io

prompt = raw_input("Enter the path for the train list\n")
train_list = io.open(prompt, 'r')
p=nb_helper2.Classifier()
for lines in train_list:
    [filepath, docClass] = lines.split()
    p.learn(filepath, docClass)

p.get_statistics()

test_path = raw_input('Enter the path for the test list\n')
test_list = io.open(test_path, 'r')
output_path = raw_input('Enter the path for the output data\n')
output  = io.open(output_path, 'wb')
for filepath in test_list:
    filepath = filepath.rstrip('\n')
    prediction = p.classify(filepath)
    print(filepath + " "  + str(prediction))
    output.write(filepath+ " " + str(prediction)+ '\n')
output.close()
