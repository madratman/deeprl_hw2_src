import numpy as np

image_1 = np.arange(9).reshape(3,3)+1
# array([[1, 2, 3],
#        [4, 5, 6],
#        [7, 8, 9]])

image_2 = image_1 + 9
# array([[10, 11, 12],
#        [13, 14, 15],
#        [16, 17, 18]])

image_3 = image_2 + 9
# array([[19, 20, 21],
#        [22, 23, 24],
#        [25, 26, 27]])

image_4 = image_3 + 9
# array([[28, 29, 30],
#        [31, 32, 33],
#        [34, 35, 36]])

history = np.zeros([4,3,3,1])

history = np.roll(history, -1, axis=0)
history[-1] = image_1[..., np.newaxis] # replaces last item by 4. => [1,2,3,4]

history = np.roll(history, -1, axis=0)
history[-1] = image_2[..., np.newaxis] 

history = np.roll(history, -1, axis=0)
history[-1] = image_3[..., np.newaxis] 

history = np.roll(history, -1, axis=0)
history[-1] = image_4[..., np.newaxis] 

# this is fed to net to predict q values
history_swap = np.swapaxes(history, 0,3) # history_swap.shape = (1, 3, 3, 4)
# array([[[[  1.,  10.,  19.,  28.],
#          [  2.,  11.,  20.,  29.],
#          [  3.,  12.,  21.,  30.]],

#         [[  4.,  13.,  22.,  31.],
#          [  5.,  14.,  23.,  32.],
#          [  6.,  15.,  24.,  33.]],

#         [[  7.,  16.,  25.,  34.],
#          [  8.,  17.,  26.,  35.],
#          [  9.,  18.,  27.,  36.]]]])

history_swap[0,...,0]
# array([[ 1.,  2.,  3.],
#        [ 4.,  5.,  6.],
#        [ 7.,  8.,  9.]])

history_swap[0,...,1]
# array([[ 10.,  11.,  12.],
#        [ 13.,  14.,  15.],
#        [ 16.,  17.,  18.]])


# in process batch, we have
# current_state_images = np.zeros([32, 84, 84, 4])
# for (idx, each_list_of_samples) in enumerate(current_state_samples):
#     current_state_images[idx, ...] = np.dstack([sample.state for sample in each_list_of_samples])

# let's verify if this is correct
history_2 = np.zeros((1,3,3,4))
history_2[0,...] = np.dstack([image_1,image_2,image_3,image_4])
# history_2
# array([[[[  1.,  10.,  19.,  28.],
#          [  2.,  11.,  20.,  29.],
#          [  3.,  12.,  21.,  30.]],

#         [[  4.,  13.,  22.,  31.],
#          [  5.,  14.,  23.,  32.],
#          [  6.,  15.,  24.,  33.]],

#         [[  7.,  16.,  25.,  34.],
#          [  8.,  17.,  26.,  35.],
#          [  9.,  18.,  27.,  36.]]]])
