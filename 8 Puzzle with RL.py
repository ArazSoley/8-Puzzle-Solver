import random
#import json

Row_Count = 4
Column_Count = 4

environment = list()

state_space = dict()

agent_position = list()

final_state_index = ''

completion_count = 35

learning_rate = 0.1

discount_rate = 0.9

prev_action = 2

distance = 0

step = 0

episode_count = 50000

for i in range(1, Row_Count * Column_Count):
    final_state_index += str(i) + ' '
final_state_index += '0 '

initial_environment_set = list(range(Row_Count * Column_Count))
random.shuffle(initial_environment_set)

# initial_environment_set = [10,7,14,2,1,9,0,4,11,6,5,12,3,15,8,13]


Goal_state_coordinates = {}

for row in range(Row_Count):
    for column in range(Column_Count):
        Goal_state_coordinates[row * Column_Count + column + 1] = [row, column]
del Goal_state_coordinates[Row_Count * Column_Count]
Goal_state_coordinates[0] = [Row_Count - 1, Column_Count - 1]


# Calculates the distance of the current state to the goal state
def Manhattan_distance():
    distance = 0
    for row in range(Row_Count):
        for column in range(Column_Count):
            abs(Goal_state_coordinates[environment[row][column]][0] - row) + \
            abs(Goal_state_coordinates[environment[row][column]][1] - column)
    return distance


# Returns true if the agent is not moving back to its previous location
def not_prev_action(action):
    if (prev_action == 0 and action != 3) or (prev_action == 1 and action != 2) or (
            prev_action == 2 and action != 1) or (prev_action == 3 and action != 0):
        return True
    return False


# Display the environment in the output
def display():
    print()
    for row in environment:
        for column in row:
            print(column, end=' ')
        print()
    print()


# Returns a reward if the current state is closer to the goal state compared to the previous state
def get_reward():
    global step
    global distance
    global completion_count

    step += 1

    if find_index() == final_state_index:
        print('got reward in', step, 'steps.')
        step = 0
        create_environment()
        completion_count -= 1
        return 10

    reward = 0

    new_distance = Manhattan_distance()

    if new_distance < distance:
        reward = 1

    distance = new_distance

    return reward


# Applies the given action to both environment and the agent_position
def act(action):
    if action == 0:

        environment[agent_position[0]][agent_position[1]], environment[agent_position[0]][agent_position[1] - 1] = \
            environment[agent_position[0]][agent_position[1] - 1], 0

        agent_position[1] -= 1

    elif action == 1:

        environment[agent_position[0]][agent_position[1]], environment[agent_position[0] - 1][agent_position[1]] = \
            environment[agent_position[0] - 1][agent_position[1]], 0

        agent_position[0] -= 1

    elif action == 2:

        environment[agent_position[0]][agent_position[1]], environment[agent_position[0] + 1][agent_position[1]] = \
            environment[agent_position[0] + 1][agent_position[1]], 0

        agent_position[0] += 1

    else:

        environment[agent_position[0]][agent_position[1]], environment[agent_position[0]][agent_position[1] + 1] = \
            environment[agent_position[0]][agent_position[1] + 1], 0

        agent_position[1] += 1


# Returns true if the agent is not moving out of the borders
def is_available(action):
    if action == 0 and agent_position[1] > 0:
        return True
    elif action == 1 and agent_position[0] > 0:
        return True
    elif action == 2 and agent_position[0] < (Row_Count - 1):
        return True
    elif action == 3 and agent_position[1] < (Column_Count - 1):
        return True
    return False


# Returns the unique index of the current state
def find_index():
    index_str = ''

    for row in range(Row_Count):
        for column in range(Column_Count):
            index_str += str(environment[row][column]) + ' '

    if not (index_str in state_space):
        state_space[index_str] = [0, 0, 0, 0]

    return index_str


# Choosing action type
def choose_action(Exploit=False):
    action_type = random.choices(['Exploit', 'Explore'], cum_weights=[3, 10])

    index = find_index()

    if action_type == 'Exploit' or Exploit:
        max_Q_value = [-1000, -1000]
        for action in range(len(state_space[index])):
            if state_space[index][action] >= max_Q_value[0] and is_available(action) and not_prev_action(action):
                max_Q_value = [state_space[index][action], action]
        action = max_Q_value[1]
    else:
        temp_action_list = [0, 1, 2, 3]
        random.shuffle(temp_action_list)

        for i in temp_action_list:
            if is_available(i) and not_prev_action(i):
                action = i
                break

    return action, index


# Creating the initial environment wrt initial_environment_set and repositioning the agent
def create_environment():
    global agent_position
    global environment

    environment = []
    temp_list = initial_environment_set[:]

    for row in range(Row_Count):
        temp = list()
        for column in range(Column_Count):
            if temp_list[0] == 0:
                agent_position = [row, column]
            temp.append(temp_list.pop(0))
        environment.append(temp)


# Getting action, performing action, and updating the Qvalues
def agent():
    global prev_action

    action, cur_index = choose_action()

    prev_action = action

    act(action)

    next_index = find_index()

    reward = get_reward()

    next_state_max_Q = -100000
    for i in state_space[next_index]:
        if i > next_state_max_Q:
            next_state_max_Q = i

    cur_state_Q_value = state_space[cur_index][action]

    state_space[cur_index][action] += learning_rate * (reward + discount_rate * next_state_max_Q + cur_state_Q_value)


# Training
create_environment()
distance = Manhattan_distance()

while completion_count > 0:
    if episode_count <= 0:
        episode_count = 50000
        create_environment()
        distance = Manhattan_distance()

    agent()
    episode_count -= 1

# Using the generated Q values to solve the puzzle step by step
create_environment()
prev_action = 2
step_count = 0
display()

while True:

    action, index = choose_action(Exploit=True)

    prev_action = action
    act(action)

    step_count += 1
    display()
    print(step_count)

    if find_index() == final_state_index:
        print('finished')
        break
