import minerl 

data = minerl.data.make(
    'MineRLObtainDiamond-v0')


for current_state, action, reward, next_state, done \
    in data.batch_iter(
        batch_size=7, num_epochs=1, seq_len=32):

        # Print the POV @ the first step of the sequence
        print(current_state['inventory'])
        print(current_state['pov'][0].shape)
        break
        # # Print the final reward pf the sequence!
        # print(reward[-1])

        # # Check if final (next_state) is terminal.
        # print(done[-1])

        # # ... do something with the data.
        # print("At the end of trajectories the length"
        #       "can be < max_sequence_len", len(reward))