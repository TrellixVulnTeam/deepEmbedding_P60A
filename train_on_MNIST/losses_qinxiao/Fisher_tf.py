import numpy as np
import tensorflow as tf
from tensorflow.contrib.losses.python.metric_learning.metric_loss_ops import pairwise_distance
def fisher_loss(labels, embeddings, margin=1.0):
    """Computes the fisher loss.    
    The loss encourages the embeddings with the same labels to be similar to each 
    other and those with different labels to be dissimilar. And we only update the 
    network weights according to the class-pair which produce the maximum loss.

    Args:
    labels: 2-D tf.int32 `Tensor` with shape [batch_size,num_class] of
        multiclass one_hot labels.
    embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
        be l2 normalized.
    margin: Float, margin term in the loss definition.
    Returns:
    triplet_loss: tf.float32 scalar.
    """
    num_of_instances_per_appeared_class=tf.reduce_sum(labels,axis=0)
    num_of_instances_per_appeared_class=tf.where(tf.not_equal(num_of_instances_per_appeared_class,0),num_of_instances_per_appeared_class,tf.ones(num_of_instances_per_appeared_class.shape)*1e-6)
    num_of_instances_per_appeared_class=tf.expand_dims(tf.reduce_sum(labels,axis=0),axis=1)
    appeared_labels=tf.cast(labels,tf.float32)
    sum_of_embeddings_per_appeared_class=tf.transpose(tf.matmul(tf.transpose(embeddings),appeared_labels))#num_of_appeared_class*num_feature
    mean_embedding_per_appeared_class=sum_of_embeddings_per_appeared_class/num_of_instances_per_appeared_class

    square_diff_per_instance=tf.square(embeddings-tf.gather(mean_embedding_per_appeared_class,tf.argmax(appeared_labels,axis=1)))
    sum_of_square_diff_per_appeared_class=tf.reduce_sum(tf.transpose(tf.matmul(tf.transpose(square_diff_per_instance),appeared_labels)),axis=1)
    sum_of_square_diff_per_appeared_class=tf.expand_dims(sum_of_square_diff_per_appeared_class,axis=1)
    var_of_embeddings_per_appeared_class=sum_of_square_diff_per_appeared_class/num_of_instances_per_appeared_class

    inter_loss_matrix=pairwise_distance(mean_embedding_per_appeared_class, squared=True)
    intra_loss_matrix=var_of_embeddings_per_appeared_class+tf.transpose(var_of_embeddings_per_appeared_class)

    loss_matrix=tf.maximum(intra_loss_matrix-inter_loss_matrix+margin,0)
    loss_matrix=loss_matrix-loss_matrix*tf.diag(tf.ones([tf.shape(loss_matrix)[0]]))

    fisher_loss=tf.reduce_max(loss_matrix)
    return fisher_loss
def main():
    inputs=np.load('inputs.npy')
    inputs=tf.convert_to_tensor(inputs,np.float32)
    targets=np.load('targets.npy')
    targets=tf.convert_to_tensor(targets)
    targets=tf.one_hot(targets,depth=4)
    loss=fisher_loss(targets,inputs,margin=1)
    print(loss.eval(session=tf.Session()))
if __name__=='__main__':
    main()