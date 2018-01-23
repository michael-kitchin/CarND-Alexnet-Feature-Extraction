import pickle
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

# Load training, validation, and test datasets via pickle.
training_file = 'input_data/train.p'
validation_file='input_data/valid.p'
testing_file = 'input_data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
# Map to convenient globals
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# Check dimension assumptions
assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))
assert(len(X_test) == len(y_test))

image_shape_raw = X_train[0].shape
arr_shape_raw = X_train.shape
set_classes = set(y_train)
nb_classes = len(set_classes)

print("Number of training examples =", len(X_train))
print("Number of validation examples =", len(X_valid))
print("Number of testing examples =", len(X_test))
print("Image data shape =", image_shape_raw)
print("Array data shape =", arr_shape_raw)
print("Number of classes =", nb_classes)

# TODO: Define placeholders and resize operation.
sign_names = pd.read_csv('signnames.csv')
batch_size = 128
keep_probability = 0.5

# x = tf.placeholder(tf.float32, (None, 32, 32, 3))
x_tensor = tf.placeholder(tf.float32, (None, 32, 32, 3))
y_tensor = tf.placeholder(tf.int32, None)
resized = tf.image.resize_images(x_tensor, (227, 227))
keep = tf.placeholder_with_default(1.0, shape=None)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)
fc_dropout = tf.nn.dropout(fc7, keep)

# TODO: Add the final layer for traffic sign classification.
shape = (fc_dropout.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
fc_new_1 = tf.Variable(tf.truncated_normal(shape=shape, mean = 0.0, stddev = 0.1))
fc_new_2 = tf.Variable(tf.zeros(nb_classes))
fc_new_3 = tf.add(tf.matmul(fc_dropout, fc_new_1),fc_new_2)
logits = tf.nn.softmax(fc_new_3)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

one_hot_y = tf.one_hot(y_tensor,nb_classes)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y,logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0013)

training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TODO: Train and evaluate the feature extraction model.
num_examples = len(X_train)

with tf.Session() as sess:
    for offset in range(0,num_examples,batch_size):
        end = offset+batch_size
        batch_x, batch_y = X_train[offset:end],y_train[offset:end]
        sess.run(training_operation,
                 feed_dict={x_tensor:batch_x,y_tensor:batch_y,keep:keep_probability})
