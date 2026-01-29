initial_learning_rate = 1.0
learning_rate_decay = 0.1
for step in range(2100):
    learning_rate = initial_learning_rate*1/((1 + (learning_rate_decay*step)))
    print(learning_rate)