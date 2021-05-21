from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop
import numpy as np
from keras.layers.advanced_activations import PReLU
import os, json

class MazeModel:

    def __init__(self, input_shape, num_actions, name= "test", model_dir="models", max_memory=10000, discount=0.95):
        self.model = self.build_model(input_shape, num_actions)
        self.name= name
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = num_actions
        self.model_dir= model_dir

    def build_model(self, input_shape, num_actions, lr=0.001):

        model = Sequential()
        model.add(Dense(input_shape, input_shape=(input_shape,)))
        model.add(PReLU())
        model.add(Dense(input_shape))
        model.add(PReLU())
        model.add(Dense(num_actions))
        model.compile(optimizer='adam', loss='mse')
        #print(model.summary())
        return model

    def remember(self, episode):
        # episode = [envstate, action, reward, envstate_next, game_over]
        # memory[i] = episode
        # envstate == flattened 1d maze cells info, including rat cell (see method: observe)
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, envstate):

        return self.model(envstate)[0]
        print("predicted sucessfully")

    def save(self):
        h5file = os.path.join(self.model_dir, self.name + ".h5")
        json_file = os.path.join(self.model_dir, self.name + ".json")
        self.model.save_weights(h5file, overwrite=True)
        with open(json_file, "w") as outfile:
            json.dump(self.model.to_json(), outfile)
        print("Saved model!")

    def load_weights(self):
        weights_file = os.path.join(self.model_dir, self.name + ".h5")
        print("loading weights from file: %s" % (weights_file,))
        self.model.load_weights(weights_file)

    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]  # envstate 1d size (1st element of episode)

        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))

        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]

            inputs[i] = envstate
            # There should be no target values for actions not taken.
            targets[i] = self.predict(envstate)
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            Q_sa = np.max(self.predict(envstate_next))
            if game_over:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets


    def train(self, data_size):
        inputs, targets = self.get_data(data_size=data_size)

        h = self.model.fit(
            inputs,
            targets,
            epochs=8,
            batch_size=16,
            verbose=0,
        )
        return h


