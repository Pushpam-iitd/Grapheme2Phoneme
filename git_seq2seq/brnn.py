import numpy as np #matrix math
import tensorflow as tf #machine learningt
import helpers #for formatting data into batches and generating random sequence data
import random
tf.reset_default_graph() #Clears the default graph stack and resets the global default graph.
sess = tf.InteractiveSession() #initializes a tensorflow session
PAD = 0
EOS = 1




file = open("cmudict.txt","r").read()


data= file.lower()
chars = list(set(data))
# data_size, vocab_size = len(data), len(chars)
# print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }
# print(ix_to_char)
words=[]
phone = []
with open("dictshuffled.txt") as f:
    for line in f:
        if "|" in line:
            param, value = line.split("|",1)
        words.append(param)
        phone.append(value)

words = [x.lower() for x in words]


pset =  set(["'"])
with open("phoneme.txt") as f:
    for line in f:
        parts = line.split()
        pset |= set(parts)
# print(pset)
# print(len(pset))
pcti = { ch:i+28 for i,ch in enumerate(sorted(pset)) }
pitc = { i+28:ch for i,ch in enumerate(sorted(pset)) }


data = open('grapheme.txt', 'r').read()
data= data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
gcti = { ch:i for i,ch in enumerate(sorted(chars)) }
gitc = { i:ch for i,ch in enumerate(sorted(chars)) }
print(gitc)
print(pitc)


# random.shuffle(words)
print("these are words:   \n",words[:100],"and the corresponding phonemes \n", phone[:100])



def getvector(word):
    tovector = []
    for c in word:
        tovector.append(gcti[c])
    return tovector

def getvector_phone(p):
    tovector = []
    parts = p.split()
    for c in parts:
        tovector.append(pcti[c])
    return tovector

# inp = []
# batches = []
# # making a batch
# for j in range (0,10000,100):
#     for i in range (j,j+100):
#         a =  getvector(words[i])
#         inp.append(a)
#     batches.append(inp)
#     inp = []
#
# print("this is input batch : \n")
# print(batches[0])



















vocab_size = 70
input_embedding_size = 150 #character length

encoder_hidden_units = 200 #num neurons
decoder_hidden_units = encoder_hidden_units * 2
#input placehodlers
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
#contains the lengths for each of the sequence in the batch, we will pad so all the same
#if you don't want to pad, check out dynamic memory networks to input variable length sequences
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)




#this thing could get huge in a real world application
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
print("this is encoder_inputs:\n",encoder_inputs)
print("this is encoder_inputs_embedded:\n",encoder_inputs_embedded)

from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple

encoder_cell  = LSTMCell(encoder_hidden_units)
((encoder_fw_outputs,
  encoder_bw_outputs),
 (encoder_fw_final_state,
  encoder_bw_final_state)) = (
    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                    cell_bw=encoder_cell,
                                    inputs=encoder_inputs_embedded,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float32, time_major=True)
    )


#Concatenates tensors along one dimension.
encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

#letters h and c are commonly used to denote "output value" and "cell state".
#http://colah.github.io/posts/2015-08-Understanding-LSTMs/
#Those tensors represent combined internal state of the cell, and should be passed together.

encoder_final_state_c = tf.concat(
    (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

encoder_final_state_h = tf.concat(
    (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

#TF Tuple used by LSTM Cells for state_size, zero_state, and output state.
encoder_final_state = LSTMStateTuple(
    c=encoder_final_state_c,
    h=encoder_final_state_h
)


decoder_cell = LSTMCell(decoder_hidden_units)
encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))
decoder_lengths = encoder_inputs_length + 3
#weights
W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
#bias
b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)

assert EOS == 1 and PAD == 0

eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

#retrieves rows of the params tensor. The behavior is similar to using indexing with arrays in numpy
eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)


#we define and return these values, no operations occur here
def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    #end of sentence
    initial_input = eos_step_embedded
    #last time steps cell state
    initial_cell_state = encoder_final_state
    #none
    initial_cell_output = None
    #none
    initial_loop_state = None  # we don't need to pass any additional information
    return (initial_elements_finished,
            initial_input,
            initial_cell_state,
            initial_cell_output,
            initial_loop_state)

def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):


    def get_next_input():
        #dot product between previous ouput and weights, then + biases
        output_logits = tf.add(tf.matmul(previous_output, W), b)
        #Logits simply means that the function operates on the unscaled output of
        #earlier layers and that the relative scale to understand the units is linear.
        #It means, in particular, the sum of the inputs may not equal 1, that the values are not probabilities
        #(you might have an input of 5).
        #prediction value at current time step

        #Returns the index with the largest value across axes of a tensor.
        prediction = tf.argmax(output_logits, axis=1)
        #embed prediction for the next input
        next_input = tf.nn.embedding_lookup(embeddings, prediction)
        return next_input


    elements_finished = (time >= decoder_lengths) # this operation produces boolean tensor of [batch_size]
                                                  # defining if corresponding sequence has ended



    #Computes the "logical and" of elements across dimensions of a tensor.
    finished = tf.reduce_all(elements_finished) # -> boolean scalar
    #Return either fn1() or fn2() based on the boolean predicate pred.
    input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)

    #set previous to current
    state = previous_state
    output = previous_output
    loop_state = None

    return (elements_finished,
            input,
            state,
            output,
            loop_state)


def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:    # time == 0
        assert previous_output is None and previous_state is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

#Creates an RNN specified by RNNCell cell and loop function loop_fn.
#This function is a more primitive version of dynamic_rnn that provides more direct access to the
#inputs each iteration. It also provides more control over when to start and finish reading the sequence,
#and what to emit for the output.
#ta = tensor array
decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack()


#to convert output to human readable prediction
#we will reshape output tensor

#Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
#reduces dimensionality
decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
#flettened output tensor
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
#pass flattened tensor through decoder
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)

# decoder_logits = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)
#prediction vals
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))

ta = tf.zeros(( 30,decoder_batch_size, vocab_size))
decoder_logits = tf.concat([decoder_logits,ta],0)


til = tf.Variable([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])
decoder_logits = tf.gather(decoder_logits,til)

print("reached here ****************************************************************************************************************************************************************")

decoder_prediction = tf.argmax(decoder_logits, 2)

stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)

#loss function
loss = tf.reduce_mean(stepwise_cross_entropy)
#train it
train_op = tf.train.GradientDescentOptimizer(9.7).minimize(loss)

sess.run(tf.global_variables_initializer())
batch_size = 100





# batches = []

max_input_size = 100000
inp = []
out = []
batches = []
label_batches = []
# making a batch
for j in range (0,max_input_size,100):
    for i in range (j,j+100):
        a =  getvector(words[i])
        b =  getvector_phone(phone[i])
        out.append(b)
        inp.append(a)
    batches.append(inp)
    label_batches.append(out)
    inp = []
    out = []

print("this is input batch : \n")
# print(batches[0])

temp1 = batches    #to print the labels
temp2 = label_batches

batches = iter(batches)
label_batches = iter(label_batches)

# batches = helpers.random_sequences(length_from=3, length_to=8,
#                                    vocab_lower=2, vocab_upper=10,
#                                    batch_size=batch_size)
# # print("this is how batches look  \n")
# # print(batches[1])
#
# print(next(batches))
# print('head of the batch:')
# for seq in next(batches):
#     print(seq)


# print("batch size is this ", tf.shape(next(batches)))

def  next_feed():
    batch = next(batches)
    label_batch = next(label_batches)
    encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
    decoder_targets_, _ = helpers.batch(
        label_batch,max_sequence_length = 30
    )
    return {
        encoder_inputs: encoder_inputs_,
        encoder_inputs_length: encoder_input_lengths_,
        decoder_targets: decoder_targets_,
    }

loss_track = []


max_batches = 1000
batches_in_epoch = 100

try:
    for batch in range(max_batches):
        fd = next_feed()
        _, l = sess.run([train_op, loss], fd)
        loss_track.append(l)

        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
            predict_ = sess.run(decoder_prediction, fd)
            for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                print('  sample {}:'.format(i + 1))
                print('    input     > {}'.format(inp))
                print('    predicted > {}'.format(pred))
                if i >= 2:
                    break
            print()

except KeyboardInterrupt:
    print('training interrupted')

# print("\n:::::::::just to see the correct outputs:::::::::\n")
# print("batch0: \n")
# print(temp1[0][0]," : ",temp2[0][0])
# print(temp1[0][1]," : ",temp2[0][1])
# print(temp1[0][2]," : ",temp2[0][2])
# print("batch100: \n")
# print(temp1[100][0]," : ",temp2[100][0])
# print(temp1[100][1]," : ",temp2[100][1])
# print(temp1[100][2]," : ",temp2[100][2])
# print("batch200: \n")
# print(temp1[200][0]," : ",temp2[200][0])
# print(temp1[200][1]," : ",temp2[200][1])
# print(temp1[200][2]," : ",temp2[200][2])
# print("batch300: \n")
# print(temp1[300][0]," : ",temp2[300][0])
# print(temp1[300][1]," : ",temp2[300][1])
# print(temp1[300][2]," : ",temp2[300][2])
# print("batch400: \n")
# print(temp1[400][0]," : ",temp2[400][0])
# print(temp1[400][1]," : ",temp2[400][1])
# print(temp1[400][2]," : ",temp2[400][2])
# print("batch500: \n")
# print(temp1[500][0]," : ",temp2[500][0])
# print(temp1[500][1]," : ",temp2[500][1])
# print(temp1[500][2]," : ",temp2[500][2])
# print("batch600: \n")
# print(temp1[600][0]," : ",temp2[600][0])
# print(temp1[600][1]," : ",temp2[600][1])
# print(temp1[600][2]," : ",temp2[600][2])
# print("batch700: \n")
# print(temp1[700][0]," : ",temp2[700][0])
# print(temp1[700][1]," : ",temp2[700][1])
# print(temp1[700][2]," : ",temp2[700][2])
# print("batch800: \n")
# print(temp1[800][0]," : ",temp2[800][0])
# print(temp1[800][1]," : ",temp2[800][1])
# print(temp1[800][2]," : ",temp2[800][2])
# print("batch900: \n")
# print(temp1[900][0]," : ",temp2[900][0])
# print(temp1[900][1]," : ",temp2[900][1])
# print(temp1[900][2]," : ",temp2[900][2])
