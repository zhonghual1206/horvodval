import numpy as np

#file = np.genfromtxt("./chat_output/test_results.tsv", delimiter='\t',dtype=np.float)
file = np.genfromtxt("deal_param_output_back/test_results.tsv", delimiter='\t',dtype=np.float)
list1 = []
print(file.shape[0])
for i in range(file.shape[0]):
    list1.append(np.argmax(file[i]))

print(list(list1))
