#%%
import tensorflow as tf
import numpy as np

#%%
@tf.function
def add1(a,b):
    print('tracing')
    tf.print('calculating')
    c = a+b
    return c


# %%
c = add1(3,4)
c = add1(4,3)
c = add1(3,4)
# %%

c4 = add1(tf.constant(3), tf.constant(4))
c4 = add1(tf.constant(4), tf.constant(3))
c4 = add1(tf.constant(3), tf.constant(4))


# %%

# %%
