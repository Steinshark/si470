import tensorflow as tf
from terminal import *

f32 = tf.dtypes.float32

holes =

s = tf.constant([[1,0,1,5,2],[0,6,1,0,2]],dtype=f32)
s = tf.sparse.from_dense(s)

d = tf.sparse.map_values(tf.ones_like,s)
print(tf.sparse.to_dense(d).numpy())
