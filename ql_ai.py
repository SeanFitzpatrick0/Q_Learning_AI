#!/usr/bin/env python3

# @author Sean Fitzpatrick 2019-3-2
# @desc Q Learing AI: This files contains the AI class trained with Q Learning and main program.
#Imports========================================
import random, copy, time, os, json, argparse

#Classes=========================================
''' @desc An AI trained using Q Learning.
    Its constructor takes learning and enviorment data for the grid game.
    The AI must be trained befor simulating a game.
'''
class Q_Learning_AI:
  
    def __init__(self, game_data):
        # Setting learing parameters
        self.num_episodes = 500
        self.max_steps_per_episode = 100
        self.learning_rate = 0.1
        self.discount_rate = 0.99
        self.exploration_rate = 1
        self.max_exploration_rate = 1
        self.min_exploration_rate = 0.01
        self.exploration_decay_rate = 0.05
        self.learning_record = []
        self.isTrained = False

        # Createing enviorment
        self.agent_token = game_data['agent_token']
        self.starting_state = tuple(game_data['starting_state'])
        self.l_state = self.starting_state
        self.token_rewards = game_data['token_rewards']
        self.board = game_data['board']
        self.q_table = self.__gen_Qtable(self.board)

    def __print_board(self):
        # Make copy of board and state
        board = copy.deepcopy(self.board)
        state = copy.deepcopy(self.l_state)

        # Add agent
        board[state[0]][state[1]] = self.agent_token

        s = ' ' + '_' * int(len(self.board[0]) * 2.9) + '\n|'
        for row in board:
            for x in row:
                s += (x + ' |')
            s += '\n|'
        s = s[:-1]      # Remove ending '|' and \n
        s += ' ' + '‾' * int(len(self.board[0]) * 2.9)
        print(s)

    def __print_qtable(self):
        for row in range(len(self.q_table)):
            for col in range(len(self.q_table[row])):
                print(self.q_table[row][col])
    
    def __gen_Qtable(self, board):
        # Initalize q_table with 0 q_values
        q_table = []
        for x in board:
            row = []
            for y in x:
                row.append([0,0,0,0])
            q_table.append(row)

        # Remove out of bounds actions
        for row in range(len(board)):
            for col in range(len(board[row])):
                if col == 0:
                    q_table[row][col][0] = None    # Cant move left
                if col == len(board[row]) - 1:
                    q_table[row][col][1] = None    # Cant move right
                if row == 0:
                    q_table[row][col][2] = None    # Cant move up
                if row == len(board) - 1:
                    q_table[row][col][3] = None    # Cant move down

        return q_table

    def __get_actions(self, state):
        actions = []
        movments = [(-1,0), (1,0), (0,-1), (0,1)]
        for move in movments:
            new_state = (state[0]+move[0], state[1]+move[1])

            # is new state in bounds of the board
            if(new_state[0] >= 0 and new_state[0] < len(self.board) and 
                new_state[1] >= 0 and new_state[1] < len(self.board[0])):
                actions.append(new_state)

        return actions

    def __calc_qvalue(self, old_value, new_state):
        # Imediate reward
        current_tile = self.board[new_state[0]][new_state[1]]

        # Find reward for token
        imediate_reward = self.token_rewards[current_tile][0]
        done = self.token_rewards[current_tile][1]
        details = self.token_rewards[current_tile][2]

        # Discounted expected value
        next_actions_values = self.q_table[self.l_state[0]][self.l_state[1]]
        next_actions_values = [x for x in next_actions_values if x is not None]   # remove None vals
        max_value = max(next_actions_values)
        discounted_expected_value = self.discount_rate * max_value

        q_value = (1 - self.learning_rate) * (old_value) + self.learning_rate * (imediate_reward + discounted_expected_value)
        return q_value, done, details

    def __get_qvalue(self, old_state, new_state):
        # Find direction moved
        movment, movment_pos = self.__get_movment(old_state, new_state)

        q_value = self.q_table[old_state[0]][old_state[1]][movment_pos]
        return q_value

    def __get_movment(self, old_state, new_state):
        # Find direction moved
        movment = (new_state[0] - old_state[0], new_state[1] - old_state[1])
        if movment == (0, -1):
            movment_pos = 0     # L
        elif movment == (0, 1):
            movment_pos = 1     # R
        elif movment == (-1, 0):
            movment_pos = 2     # U
        else:
            movment_pos = 3     # D
        
        return movment, movment_pos

    def __transition(self, new_state):
        # Get old q-value
        old_state = self.l_state
        movment, movment_pos = self.__get_movment(old_state, new_state)
        old_value = self.q_table[self.l_state[0]][self.l_state[1]][movment_pos]

        # Move agent to new position
        self.l_state = new_state

        # Update Q-table
        q_value, done, details = self.__calc_qvalue(old_value, new_state)
        self.q_table[old_state[0]][old_state[1]][movment_pos] = q_value

        return q_value, done, details

    def learn(self, display_results=False):
        rewards_all_episodes = []

        for episode in range(self.num_episodes):
            # set starting episode values
            self.l_state = self.starting_state
            rewards_current_episode = 0
            done = False

            for step in range(self.max_steps_per_episode):

                # explore or exploit
                exploration_rate_threshold = random.uniform(0, 1)
                if exploration_rate_threshold > self.exploration_rate:
                    # preform the best known action
                    actions = self.__get_actions(self.l_state)
                    act_qval = {x : self.__get_qvalue(self.l_state, x) for x in actions}
                    action = max(act_qval, key=act_qval.get)
                    reward, done, _ = self.__transition(action)
                    rewards_current_episode += reward
                else:
                    # preform random action
                    actions = self.__get_actions(self.l_state)
                    action = random.choice(actions)
                    reward, done, _ = self.__transition(action)
                    rewards_current_episode += reward

                if done:
                    break

            # add episode reward
            rewards_all_episodes.append(rewards_current_episode)

            # record episode variables
            self.learning_record.append({
                'Episode_Reward' : rewards_current_episode,
                'Exploration_Rate' : self.exploration_rate,
                'Q_Table' : self.q_table
            })

            # decay exploration rate
            self.exploration_rate *= (1 - self.exploration_decay_rate)

        self.isTrained = True
        # @opt Print learning results
        if display_results:
            print("******** Reward per episodes ********")
            for i in range(len(rewards_all_episodes)):
                print('%d : %f' % (i, rewards_all_episodes[i])) 
        
    def simulate(self):
        assert self.isTrained, 'Error: AI must be trained before simulating'

        os.system('clear')
        q_table = self.learning_record[-1]['Q_Table']
        self.l_state = self.starting_state

        for step in range(self.max_steps_per_episode):
            self.__print_board()
            # preform the best known action
            actions = self.__get_actions(self.l_state)
            act_qval = {x : self.__get_qvalue(self.l_state, x) for x in actions}
            action = max(act_qval, key=act_qval.get)
            reward, done, details = self.__transition(action)

            # wait between moves
            time.sleep(0.5)
            # clear previous state
            os.system('clear')
            
            if done:
                break

        # Display end state
        self.__print_board()
        if details == None:
            print('OUT OF STEPS')
        else:
            print(details)

#Functions=======================================
def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("Error: The file %s does not exist" % arg)
    else:
        return open(arg, 'r')  # return an open file handle

#Program=========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='An AI trained using Q Learning that simulates '\
        'games that are defined in a template JSON file')
    parser.add_argument('-t', '--template', help='The template file path for the selected game',
        dest='template_file', metavar='FILE', required=True)
    args = parser.parse_args()

    # Load template
    try:
        json_file = open(args.template_file)
        json_str = json_file.read()
        game_data = json.loads(json_str)
    except IOError:
        print('Error: The file %s does not exist' % args.template_file)
        exit()

    #Create AI
    ai = Q_Learning_AI(game_data)
    ai.learn()
    ai.simulate()
