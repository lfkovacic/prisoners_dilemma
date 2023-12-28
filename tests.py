import tensorflow as tf
import numpy as np

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training parameters
epochs = 10
games_per_epoch = 1

for epoch in range(epochs):
    # Collect data for training from self-play games
    inputs = []
    labels = []

    for _ in range(games_per_epoch):
        # Initialize the game state (last 8 decisions of the opponent)
        game_state = np.zeros((1, 8))

        # Play the game for a fixed number of steps
        for _ in range(10):  # You can adjust the number of steps per game
            # Get the model's decision based on the current game state
            decision = model.predict(game_state)

            # Apply a threshold to get the binary decision (0 or 1)
            binary_decision = 1 if decision >= 0.5 else 0

            # Update the game state with the model's decision
            game_state = np.concatenate([game_state[:, 1:], [[binary_decision]]], axis=1)

        # Assign a reward based on the game outcome (you need to define this based on the rules)
        reward = 0 if np.sum(game_state) <= 3 else 1

        # Append the game state and the corresponding reward to the training data
        inputs.append(np.squeeze(game_state))
        labels.append(reward)

    inputs = np.array(inputs)
    labels = np.array(labels)

    # Train the model on the collected data
    model.fit(inputs, labels, epochs=1, verbose=0)

    # Print the average reward for monitoring progress
    #avg_reward = np.mean(labels)
    #print(f"Epoch: {epoch + 1}, Average Reward: {avg_reward}")

# Test the final model
test_input = np.array([[0,0,0,0,0,0,0,0]])
predicted_output = model.predict(test_input)
predicted_decision = 1 if predicted_output[0, 0] >= 0.5 else 0

print("Final Predicted Decision:", predicted_decision)
# Test the final model
test_input = np.array([[1,1,1,1,1,1,1,1]])
predicted_output = model.predict(test_input)
predicted_decision = 1 if predicted_output[0, 0] >= 0.5 else 0
print("Final Predicted Decision:", predicted_decision)

# Test the final model
test_input = np.array([[1,1,1,1,0,0,0,0]])
predicted_output = model.predict(test_input)
predicted_decision = 1 if predicted_output[0, 0] >= 0.5 else 0

print("Final Predicted Decision:", predicted_decision)