import tensorflow as tf

# training data
x_train = [1, 2, 3, 4]
y_train = [1, 2, 3, 4]

# tf Graph Input
X = tf.placeholder(tf.float32,  name="input")
Y = tf.placeholder(tf.float32)

# Set model weights
M = tf.Variable(.3, dtype=tf.float32)
c = tf.Variable(-.3, dtype=tf.float32)

# Construct a linear model
pred = tf.add(tf.multiply(X, M), c, name="pred")

# Mean squared error
loss = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*len(x_train))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {X: x_train, Y: y_train})

builder = tf.saved_model.builder.SavedModelBuilder("linearmodel")

builder.add_meta_graph_and_variables(sess,["serve"])

builder.save()

