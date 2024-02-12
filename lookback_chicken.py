from q_learning import QLearning
from agt_server.local_games.chicken_arena import ChickenArena
from agt_server.agents.test_agents.chicken.basic_agent.my_agent import BasicAgent

class LookBackChicken(QLearning):
    def __init__(self, name, num_possible_states, num_possible_actions, initial_state, learning_rate, discount_factor, exploration_rate, training_mode, save_path=None) -> None:
        super().__init__(name, num_possible_states, num_possible_actions, initial_state,
                         learning_rate, discount_factor, exploration_rate, training_mode, save_path)

    def determine_state(self):
        opp_action_hist = self.get_opp_action_history()
        last_action = opp_action_hist[-1]
        if (len(opp_action_hist) < 2):
            return last_action

        second_last_action = opp_action_hist[-2]
        return 2 * second_last_action + last_action


NUM_TRAINING_ITERATIONS = 20000
NUM_ITERATIONS_PER_PRINT = 1000
# this agent only uses 4 states
NUM_POSSIBLE_STATES = 4
# chicken has 2 possible actions (CONTINUE and SWERVE).
NUM_POSSIBLE_ACTIONS = 2
INITIAL_STATE = 0

LEARNING_RATE = 0.05
DISCOUNT_FACTOR = 0.90
EXPLORATION_RATE = 0.05

if __name__ == "__main__":

    #### DO NOT TOUCH THIS #####
    name = "Last Move Chicken"
    train = True
    save_q_table = None # Set to None to not save the q-table, otherwise save it under the file path that you give it E.g. "qtable.npy" 
                        # If the file already exists then it will initialize the q-table using the saved npy file. 


    # START SIMULATING THE GAME
    agent = LookBackChicken(name, NUM_POSSIBLE_STATES, NUM_POSSIBLE_ACTIONS,
                            INITIAL_STATE, LEARNING_RATE, DISCOUNT_FACTOR, EXPLORATION_RATE, train, save_q_table)
    print("TRAINING")
    arena = ChickenArena(
        num_rounds=20000,
        timeout=1,
        players=[
            agent,
            BasicAgent("Agent_1"),
            BasicAgent("Agent_2"),
            BasicAgent("Agent_3"),
            BasicAgent("Agent_4")
        ]
    )
    arena.run()
    print("TESTING")
    agent.set_training_mode(False)
    arena = ChickenArena(
        num_rounds=1000,
        timeout=1,
        players=[
            agent,
            BasicAgent("Agent_1"),
            BasicAgent("Agent_2"),
            BasicAgent("Agent_3"),
            BasicAgent("Agent_4")
        ]
    )
    arena.run()
