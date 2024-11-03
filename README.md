# Self-Driving Car with PPO in CarRacing-v2

This project explores training a self-driving car agent in the CarRacing-v2 environment from OpenAI Gym using reinforcement learning. The agent is trained with the Proximal Policy Optimization (PPO) algorithm, leveraging the default convolutional neural network (CNN) policy provided by the Stable-Baselines3 library.

## Project Overview

The goal of this project is to enable a virtual car agent to autonomously navigate a racing track, achieving high rewards by successfully completing laps and avoiding collisions. Using PPO, a widely-used reinforcement learning algorithm, the agent learns through trial and error to improve its driving skills in a simulated environment.

## Key Features

- **Algorithm**: Proximal Policy Optimization (PPO) with Stable-Baselines3’s default CnnPolicy.
- **Environment**: CarRacing-v2, providing a continuous control challenge with complex tracks.
- **Training and Evaluation**: The model is trained for 100,000 timesteps, followed by evaluation to assess its performance.

## Setup

### Dependencies

Install the required libraries:

```bash
pip install gym[box2d] stable-baselines3 pyglet
```

### File Structure

- **train.py**: Main script for training the PPO agent in the CarRacing-v2 environment.
- **evaluate.py**: Evaluates the trained model over multiple episodes.
- **Trained Model Checkpoints**: Saved in `Training/Saved Models/` for future use.

## Usage

### Training the Agent

To train the agent, run:

```python
python train.py
```

This script sets up the CarRacing-v2 environment and uses the PPO algorithm to train the agent. The training logs can be viewed in TensorBoard.

### Evaluating the Agent

To evaluate the model’s performance, run:

```python
python evaluate.py
```

This will test the trained agent over a series of episodes and display its performance.

## Results

After training, the model can complete laps autonomously, demonstrating learned behavior for cornering, speed control, and avoiding obstacles. Future enhancements could involve experimenting with custom neural networks for more advanced behavior.
