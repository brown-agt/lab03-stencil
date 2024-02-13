# CS1440/2440 Lab 3: QLearning and Collusion

## Introduction
Please implement Q-learning in two simulated environments. The lab emphasizes the importance of state representation in RL and demonstrates that Q-learners can learn collusive strategies in competitive games.

## Setup and Installation
Follow these steps to set up your environment and install the necessary package for the lab.

### Step 1: Git Clone the Repository 
Open your terminal and navigate to where you want to clone the repository
```bash 
git clone https://github.com/brown-agt/lab03-stencil.git
```

### Step 2: Create a Virtual Environment
Please then navigate to your project directory. Run the following commands to create a Python virtual environment named `.venv`.

If you own a Mac 
```bash
python -m venv .venv
source .venv/bin/activate
```

If you own a Windows 
```bash 
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install the agt server package
```bash
pip install --upgrade pip
pip install --upgrade agt-server
```

## Agent Methods 
For the `ChickenAgent`s here are a few methods that you may find helpful! 
- `self.calculate_utils(a1, a2)` is a method that takes in player 1's action (`a1`) and player 2's action (`a2`) and returns a list [`u1`, `u2`] where `u1` is player1's utility and `u2` is player 2's utility. 
- `self.get_action_history()` is a method that returns a list of your actions from previous rounds played.
- `self.get_util_history()` is a method that returns a list of your utility from previous rounds played. 
- `self.get_opp_action_history()` is a method that returns a list of your opponent's actions from previous rounds played.
- `self.get_opp_util_history()` is a method that returns a list of your opponent's utility from previous rounds played.
- `self.get_last_action()` is a method that returns a your last action from the previous round.
- `self.get_last_util()` is a method that returns a your last utility from the previous round.
- `self.get_opp_action_history()` is a method that returns a your opponent's last action from the previous round.
- `self.get_opp_util_history()` is a method that returns a your opponent's last utility from the previous round.
