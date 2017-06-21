from EUNN import *

import warnings
from heapq import merge

# generate rotation pairs to parameterize n x m semi-orthogonal matrix with input m-vector and output n-vector
def get_rotations(n, m):
    def pairs(x):
        return zip(*[iter(x)]*2) # reshapes list to pairs (dropping last element if necessary)

    # need to reverse order of rotations for matrices n > m
    reverse = True
    if n < m:
        # otherwise, compute with swapped variables and don't reverse
        n, m = m, n
        reverse = False

    rotation_pairs_list = []
    column_ind_list = [range(n)] # initialize to the first column of matrix
    for i in range(m-1):
        column_ind_list.append([])

    while True:
        finished_columns = 0
        rotation_pairs = []
        # print(column_ind_list) # debug

        new_column_ind_list = [column_ind_list[0][::2]]
        for i in range(m):
            # appends new rotations to list if necessary
            if len(column_ind_list[i]) > 1:
                rotation_pairs.extend(pairs(column_ind_list[i]))

                if i < m-1:
                    # in next column i+1, adds new indexes for rotation while removing rotated indexes
                    new_column_ind_list.append(list(merge(column_ind_list[i+1][::2], column_ind_list[i][1::2])))
            else:
                # if no new rotations in column i, then just append for next column i+1
                if i < m-1:
                    new_column_ind_list.append(column_ind_list[i+1][::2])

        column_ind_list = new_column_ind_list
        if len(rotation_pairs) > 0:
            rotation_pairs_list.append(rotation_pairs)
        else: # if no new rotation pairs, then we are done
            break

    if reverse:
        rotation_pairs_list.reverse()
    return rotation_pairs_list

# generate rotation pairs to parameterize n x m semi-orthogonal matrix with input m-vector and output n-vector
def get_rotations_pretty(n, m):
    def pairs(x):
        return zip(iter(x[:int(len(x)/2)]),iter(x[int((len(x)+1)/2):])) # reshapes list to pairs (dropping last element if necessary)

    # need to reverse order of rotations for matrices n > m
    reverse = True
    if n < m:
        # otherwise, compute with swapped variables and don't reverse
        n, m = m, n
        reverse = False

    rotation_pairs_list = []
    column_ind_list = [range(n)] # initialize to the first column of matrix
    for i in range(m-1):
        column_ind_list.append([])

    while True:
        finished_columns = 0
        rotation_pairs = []
        # print(column_ind_list) # debug

        new_column_ind_list = [column_ind_list[0][:int((len(column_ind_list[0])+1)/2)]]
        for i in range(m):
            # appends new rotations to list if necessary
            if len(column_ind_list[i]) > 1:
                rotation_pairs.extend(pairs(column_ind_list[i]))

                if i < m-1:
                    # in next column i+1, adds new indexes for rotation while removing rotated indexes
                    new_column_ind_list.append(list(merge(column_ind_list[i+1][:int((len(column_ind_list[i+1])+1)/2)]
                                                        , column_ind_list[i][int((len(column_ind_list[i])+1)/2):])))
            else:
                # if no new rotations in column i, then just append for next column i+1
                if i < m-1:
                    new_column_ind_list.append(column_ind_list[i+1][:int((len(column_ind_list[i+1])+1)/2)])

        column_ind_list = new_column_ind_list
        if len(rotation_pairs) > 0:
            rotation_pairs_list.append(rotation_pairs)
        else: # if no new rotation pairs, then we are done
            break

    if reverse:
        rotation_pairs_list.reverse()
    return rotation_pairs_list

# # test get_rotations
# rotations = get_rotations_pretty(32,64)
# print(rotations)
# print([len(pairs) for pairs in rotations])
# # print(sum([len(pairs) for pairs in rotations]))
# print(len(rotations))

def permute_rotation_pairs(hidden_size, rotation_pairs):
    num_rotations = len(rotation_pairs)

    ind = list(range(hidden_size))
    ind1 = list(range(2*num_rotations, hidden_size))
    ind2 = [-1] * hidden_size
    
    for i in range(num_rotations):
        ind[rotation_pairs[i][0]], ind[rotation_pairs[i][1]] = ind[rotation_pairs[i][1]], ind[rotation_pairs[i][0]]
        ind2[rotation_pairs[i][0]] = i
        ind2[rotation_pairs[i][1]] = i + num_rotations

    for i in range(hidden_size):
        if ind2[i] == -1:
            ind2[i] = ind1.pop()

    return ind, ind2

# generate v1, v2, ind for one disjoint set of rotation pairs, hidden_size is the larger dimension of matrix
def get_single_rotation_params(hidden_size, rotation_pairs, theta_phi_initializer):
    num_rotations = len(rotation_pairs)

    params_theta = vs.get_variable("theta", num_rotations, initializer=theta_phi_initializer)

    cos_theta = math_ops.cos(params_theta)
    sin_theta = math_ops.sin(params_theta)

    cos_list = array_ops.concat([cos_theta, cos_theta, np.ones(hidden_size-2*num_rotations)], 0)
    sin_list = array_ops.concat([sin_theta, -sin_theta, np.zeros(hidden_size-2*num_rotations)], 0)

    ind, ind2 = permute_rotation_pairs(hidden_size, rotation_pairs)

    v1 = permute(cos_list, ind2)
    v2 = permute(sin_list, ind2)

    return v1, v2, ind

def get_rotation_params(hidden_size, capacity, rotation_pairs_list):
    if capacity == 0:
        capacity = len(rotation_pairs_list)
    elif capacity < 0 or capacity > len(rotation_pairs_list):
        raise ValueError("capacity = {} is invalid (maximum capacity = {}). Set capacity = 0 to automatically use maximum capacity.".format(capacity, len(rotation_pairs_list)))

    theta_phi_initializer = init_ops.random_uniform_initializer(-np.pi, np.pi)
    # theta_phi_initializer = init_ops.random_normal_initializer(0,0.1)

    v1 = []
    v2 = []
    ind = []

    for i in range(capacity):
        with tf.variable_scope(str(i)):
            v1_tmp, v2_tmp, ind_tmp = \
                get_single_rotation_params(hidden_size, rotation_pairs_list[i], theta_phi_initializer)
        v1.append(v1_tmp)
        v2.append(v2_tmp)
        ind.append(ind_tmp)

    v1 = toTensorArray(v1)
    v2 = toTensorArray(v2)
    ind = toTensorArray(np.array(ind).astype(np.int32))

    diag = None

    return v1, v2, ind, diag, capacity


# # test get params
# rotations = [[(0, 10), (1, 11), (2, 12), (3, 13), (4, 14), (5, 15), (6, 16), (7, \
#             17), (8, 18), (9, 19)], [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (10, \
#             15), (11, 16), (12, 17), (13, 18), (14, 19)], [(0, 3), (1, 4), (5, \
#             10), (6, 11), (7, 12), (8, 13), (9, 14)], [(0, 2), (3, 7), (4, 8), \
#             (5, 9)], [(0, 1), (2, 5), (3, 6)], [(1, 3), (2, 4)], [(1, 2)]]

# # print(get_single_rotation_params(20, rotations))
# print(get_rotation_params(20, 0, rotations))

def EUNN_rect(input, dim, capacity=0, comp=False):

    height = dim[0]
    width = dim[1]
    assert height == int(input.get_shape()[-1])

    rotation_pairs_list = get_rotations_pretty(width, height) # height and width swapped since EUNN_loop acts on row vectors
    v1, v2, ind, diag, capacity = get_rotation_params(max(dim), capacity, rotation_pairs_list)

    # if height > width, project matrix at the end
    if height > width:
        output = array_ops.slice(EUNN_loop(input, capacity, v1, v2, ind, diag), [0, 0], dim)
    # if width > height, project matrix at the beginning, i.e. equivalent to padding input
    else:
        if height == width:
            warnings.warn("Consider using EUNN instead of EUNN_rect for square unitary matrices.")

        output = EUNN_loop(array_ops.pad(input,((0, 0), (0, width - height))), capacity, v1, v2, ind, diag)

    return output

# # test EUNN_rect
# EUNN_rect(ops.convert_to_tensor(np.eye(36), dtype=tf.float32), width=32)

# total_parameters = 0
# for variable in tf.trainable_variables():
#     # shape is an array of tf.Dimension
#     shape = variable.get_shape()
#     # print(shape)
#     # print(len(shape))
#     variable_parameters = 1
#     for dim in shape:
#         # print(dim)
#         variable_parameters *= dim.value
#     # print(variable_parameters)
#     total_parameters += variable_parameters
# print(total_parameters)









