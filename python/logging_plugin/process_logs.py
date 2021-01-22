log_file = 'log_example.txt'

with open(log_file, 'r') as f:
    logs = f.readlines()

user_commands, user_blocks_placed, user_blocks_removed = 0, 0, 0
chat_history = []

for log in logs:
    if 'CHAT 1' in log:
        if ('yes' not in log) and ('hello' not in log):
            user_commands += 1
    elif 'PLAYER_BROKEN_BLOCK' in log:
        user_blocks_removed += 1
    elif 'PLAYER_PLACED_BLOCK' in log:
        user_blocks_placed += 1
    elif 'CHAT 29' in log:
        if 'sorry' in log:
            chat_history.append('unparsable')
        elif 'from user generated command' in log:
            chat_history.append('induced')
        else:
            chat_history.append('core')

import pdb; pdb.set_trace()

