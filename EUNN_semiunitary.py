from EUNN import *

import warnings
from heapq import merge

# Generate rotation pairs to parameterize n x m semi-orthogonal matrix with input m-vector and output n-vector
def get_rotations_pretty(n, m, use_hybrid_method=True):
    def pairs(x):
        return zip(iter(x[:int(len(x)/2)]),iter(x[int((len(x)+1)/2):])) # reshapes list to pairs (dropping last element if necessary)

    # need to reverse order of rotations for matrices n > m
    reverse = True
    if n < m:
        # otherwise, compute with swapped variables and don't reverse
        n, m = m, n
        reverse = False

    if m == 1: # don't use hybrid method for m = 1
        use_hybrid_method = False

    if use_hybrid_method:
        rotation_pairs_list = []
        column_ind_list = [range(m, n)] # initialize to the first column of matrix
        for i in range(m-1):
            column_ind_list.append([])
    else:
        rotation_pairs_list = []
        column_ind_list = [range(n)] # initialize to the first column of matrix
        for i in range(m-1):
            column_ind_list.append([])

    while True:
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
        elif 0 not in column_ind_list[0]:
            rotation_pairs_list.append([])
        else: # if not in delay period and no new rotation pairs, then we are done
            break

        if use_hybrid_method:
            # instead of inserting the diagonal elements into the greedy algorithm earlier,
            # insert them with a delay to allow the square rotations to fit better
            current_col = len(rotation_pairs_list) - 1
            if current_col < m:
               column_ind_list[current_col].insert(0, current_col)

    if use_hybrid_method:
        rotation_pairs_list = add_square_rotations(rotation_pairs_list, n, m)

    if reverse:
        rotation_pairs_list.reverse()
    return rotation_pairs_list

# Parameterizes the square part of the rectangular matrix using both row and column rotations
def add_square_rotations(rotation_pairs_list, n, m):
    def get_square_rotations(m):
        def pairs(x):
            return zip(*[x]*2) # reshapes list to pairs (dropping last element if necessary)

        in_rotation_list = [pairs(iter(range(i, m))) for i in range(m-1)]
        out_rotation_list = [pairs(reversed(range(i))) for i in reversed(range(2, m))]

        return in_rotation_list, out_rotation_list

    if m > 1:
        if n > m:
            in_rotation_list, out_rotation_list = get_square_rotations(m)

            # merge square row rotations
            for i in range(len(in_rotation_list)):
                rotation_pairs_list[i].extend(in_rotation_list[i])

            # merge square column rotations
            capacity = len(rotation_pairs_list)
            for i in range(len(out_rotation_list)):
                rotation_pairs_list[capacity-1-i].extend(out_rotation_list[i])

        # if matrix is square, use optimal method
        elif n == m:
            def pairs(x):
                return zip(*[iter(x)]*2) # reshapes list to pairs (dropping last element if necessary)

            # capacity = m
            for i in range(m):
                rotation_pairs_list = [pairs(range(m)), pairs(range(1,m))] * int((m+1)/2)
                rotation_pairs_list = rotation_pairs_list[:m]


    return rotation_pairs_list

# Generates permutation ind (used in applying Givens rotations)
# and permutation ind2 (to map cos_list, sin_list to proper locations in v1, v2)
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

# Generate v1, v2, ind for one disjoint set of rotation pairs, hidden_size is the larger dimension of matrix
def get_single_rotation_params(hidden_size, rotation_pairs, theta_phi_initializer):
    num_rotations = len(rotation_pairs)

    params_theta = vs.get_variable("theta", num_rotations, initializer=theta_phi_initializer)

    cos_theta = math_ops.cos(params_theta)
    sin_theta = math_ops.sin(params_theta)

    cos_list = array_ops.concat([cos_theta, cos_theta, np.ones(hidden_size-2*num_rotations)], 0)
    sin_list = array_ops.concat([sin_theta, -sin_theta, np.zeros(hidden_size-2*num_rotations)], 0)

    ind, ind2 = permute_rotation_pairs(hidden_size, rotation_pairs)

    v1 = tf.gather(cos_list, ind2)
    v2 = tf.gather(sin_list, ind2)

    return v1, v2, ind

# Generates list of v1, v2, ind for all rotation pairs in list
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

def gather_cols(params, indices, name=None):
    """Gather columns of a 2D tensor.

    Args:
        params: A 2D tensor.
        indices: A 1D tensor. Must be one of the following types: ``int32``, ``int64``.
        name: A name for the operation (optional).

    Returns:
        A 2D Tensor. Has the same type as ``params``.
    """
    with tf.name_scope(name, "gather_cols", [params, indices]) as scope:
        # Check input
        params = tf.convert_to_tensor(params, name="params")
        indices = tf.convert_to_tensor(indices, name="indices")
        try:
            params.get_shape().assert_has_rank(2)
        except ValueError:
            raise ValueError('\'params\' must be 2D.')
        try:
            indices.get_shape().assert_has_rank(1)
        except ValueError:
            raise ValueError('\'params\' must be 1D.')

        # Define op
        p_shape = tf.shape(params)
        p_flat = tf.reshape(params, [-1])
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],
                                       [-1, 1]) + indices, [-1])
        return tf.reshape(tf.gather(p_flat, i_flat),
                          [p_shape[0], -1])

def EUNN_loop_gather_cols(h, L, v1_list, v2_list, ind_list, D):

    i = 0

    def F(x, i):


        v1 = v1_list.read(i)
        v2 = v2_list.read(i)
        ind = ind_list.read(i)

        diag = math_ops.multiply(x, v1)
        off = math_ops.multiply(x, v2)
        Fx = diag + gather_cols(off, ind)

        i += 1

        return Fx, i

    def cond(x, i):
        return i < L

    loop_vars = [h, i]
    FFx, _ = control_flow_ops.while_loop(
        cond, 
        F, 
        loop_vars
    )

    if not D == None:
        Wx = math_ops.multiply(FFx, D)
    else:
        Wx = FFx

    return Wx

# Multiplies by semi-unitary matrix using Givens rotations
def EUNN_rect(input, dim, capacity=0, comp=False, use_hybrid_method=True, use_gather_cols=False):
    height = dim[0]
    width = dim[1]
    assert height == int(input.get_shape()[-1])

    # height and width swapped since EUNN_loop acts on row vectors
    rotation_pairs_list = get_rotations_pretty(width, height, use_hybrid_method)

    v1, v2, ind, diag, capacity = get_rotation_params(max(dim), capacity, rotation_pairs_list)

    # if height > width, project matrix at the end
    if height > width:
        if use_gather_cols:
            output = array_ops.slice(EUNN_loop_gather_cols(input, capacity, v1, v2, ind, diag), [0, 0], [-1, width])
        else:
            output = array_ops.slice(EUNN_loop(input, capacity, v1, v2, ind, diag), [0, 0], [-1, width])

    # if width > height, project matrix at the beginning, i.e. equivalent to padding input
    else:
        if height == width:
            warnings.warn("Consider using EUNN instead of EUNN_rect for square unitary matrices.")

        if use_gather_cols:
            output = EUNN_loop_gather_cols(array_ops.pad(input,((0, 0), (0, width - height))), capacity, v1, v2, ind, diag)
        else:
            output = EUNN_loop(array_ops.pad(input,((0, 0), (0, width - height))), capacity, v1, v2, ind, diag)

    return output

# # test EUNN_rect
# EUNN_rect(ops.convert_to_tensor(np.eye(36), dtype=tf.float32), [36, 32])

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









