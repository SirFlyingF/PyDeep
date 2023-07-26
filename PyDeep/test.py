import numpy as np
import nnet
import activations as acts
from random import randint

train_samples = []
train_labels = []

for i in range(50):
	random_young = randint(13,64)
	train_samples.append(random_young)
	train_labels.append(1)

	random_old = randint(65,100)
	train_samples.append(random_old)
	train_labels.append(0)

for i in range(1000):
	random_young = randint(13, 64)
	train_samples.append(random_young)
	train_labels.append(0)

	random_old = randint(65,100)
	train_samples.append(random_old)
	train_labels.append(1)

test_samples = []
test_labels = []

for i in range(5):
	random_young = randint(13,64)
	test_samples.append(random_young)
	test_labels.append(1)

	random_old = randint(65,100)
	test_samples.append(random_old)
	test_labels.append(0)

for i in range(100):
	random_young = randint(13, 64)
	test_samples.append(random_young)
	test_labels.append(0)

	random_old = randint(65,100)
	test_samples.append(random_old)
	test_labels.append(1)

train_samples = np.array(train_samples)
train_labels = np.array(train_labels)

acts = [acts.relu(), acts.relu(), acts.sigmoid()]
net = nnet.mlp(dims=[1, 3, 1], acts=acts)

net.train(learning_rate=0.01, train_set=train_samples, labels=train_labels, epoch=100, batch_size=10, plot=True, verbose=True)

perc = net.test(test_set=test_samples, labels=test_samples, pred=False)
print(f"Accuracy {perc}%")
