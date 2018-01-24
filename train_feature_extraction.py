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
learning_rate = 0.0013

# x = tf.placeholder(tf.float32, (None, 32, 32, 3))
images_tensor = tf.placeholder(tf.float32, (None, 32, 32, 3))
labels_tensor = tf.placeholder(tf.int64, None)
resized = tf.image.resize_images(images_tensor, (227, 227))
keep = tf.placeholder_with_default(1.0, shape=None)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)
fc7_dropout = tf.nn.dropout(fc7, keep)

# TODO: Add the final layer for traffic sign classification.
fc7_shape = (fc7_dropout.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
fc_new_W = tf.Variable(tf.truncated_normal(shape=fc7_shape, mean = 0.0, stddev = 0.1))
fc_new_b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7_dropout, fc_new_W, fc_new_b)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy_operation = tf.nn.softmax_cross_entropy_with_logits(labels=labels_tensor, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy_operation)
optimizer_operation = tf.train.AdamOptimizer(learning_rate=learning_rate)

training_operation = optimizer_operation.minimize(loss_operation, var_list=[fc_new_W, fc_new_b])
init_operation = tf.initialize_all_variables()

prediction_operation = tf.argmax(logits, 1)
accuracy_operation = tf.reduce_mean(tf.cast(tf.equal(prediction_operation, labels_tensor), tf.float32))

# TODO: Train and evaluate the feature extraction model.
num_examples = len(X_train)
with tf.Session() as sess:
    sess.run(init_operation)
    for offset in range(0,num_examples,batch_size):
        end = offset+batch_size
        batch_x, batch_y = X_train[offset:end],y_train[offset:end]
        sess.run(training_operation,
                 feed_dict={images_tensor:batch_x, labels_tensor:batch_y, keep:keep_probability})
