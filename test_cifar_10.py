import theano.tensor as T
import lasagne
import theano
import numpy as np
import sys,os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

# create batches
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# extract data
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# prepares data sets
def convert_data(data_vectorized):
    data_train = np.zeros([data_vectorized.shape[0],3,32,32],np.float32)
    for i in range(data_vectorized.shape[0]):
        data_tmp = np.moveaxis(np.reshape(data_vectorized[i],(32,32,3),'F')[:,:,::-1],2,0)
        for channel in range(3):
            data_train[i,channel,:,:] = data_tmp[channel,:,:].T/255.0
    return data_train

# prepares data labels
def convert_label(label_chiffre):
    labs = np.zeros([len(label_chiffre),10],np.float32)
    for i in range(len(label_chiffre)):
        labs[i][label_chiffre[i]]=1
    return labs

# Renvoi le pourcentage de predictions reussies
def get_pourcentage_guess(predictions, verite):
    nb_predits,nb_reussi = 0,0
    for i in range(predictions.shape[0]):
        chiffre_predit = np.argmax(predictions[i])
        chiffre_verite = np.argmax(verite[i])
        #COMPARER CES CHIFFRES
        nb_predits+=1
        if chiffre_predit == chiffre_verite:
            nb_reussi+=1
    return (nb_reussi/nb_predits)*100


data = unpickle('/home/oukache-a/Documents/stage/cifar-10-batches-py/data_batch_1')

# --------- Partitionnement des donnees (train, validation, test)

# Recuperation des donnees de training
data_train = convert_data(data[b'data'][:700])
label_train = convert_label(data[b'labels'][:700])

# Recuperation des donnees de validation
data_val = convert_data(data[b'data'][7000:9000])
label_val = convert_label(data[b'labels'][7000:9000])

# Recuperation des donnees de test
data_test = convert_data(data[b'data'][9000:])
label_test = convert_label(data[b'labels'][9000:])

# Declaration des variables theano
donnees_entree = T.tensor4('donnees_entrees', 'float32')
labels = T.matrix('labels', 'float32')

# -------- Creation du reseau de neurones

# Creation de la couche d'entree
input_layer = lasagne.layers.InputLayer(shape=(None,3,32,32),input_var=donnees_entree)

# Creation des couches intermediaires
inter_conv_layer_1 = lasagne.layers.Conv2DLayer(input_layer, num_filters=64,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify)
inter_pool_layer_2 = lasagne.layers.Pool2DLayer(inter_conv_layer_1,pool_size=2)

inter_conv_layer_3 = lasagne.layers.Conv2DLayer(inter_pool_layer_2, num_filters=32,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify)
inter_pool_layer_4 = lasagne.layers.Pool2DLayer(inter_conv_layer_3,pool_size=2)


inter_dense_layer_5 = lasagne.layers.DenseLayer(inter_pool_layer_4,num_units=600)

# Creation de la couche de sortie (10 classes)
output_layer = lasagne.layers.DenseLayer(inter_dense_layer_5, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)

# Output des predictions
predictions = lasagne.layers.get_output(output_layer)

# Calcul de l'erreur
erreur = T.mean(lasagne.objectives.categorical_crossentropy(predictions,labels))
parametres = lasagne.layers.get_all_params(output_layer,trainable=True)
mis_a_jour = lasagne.updates.nesterov_momentum(erreur,parametres,5e-3)

# Creation des funtions theano
f_train = theano.function([donnees_entree,labels],erreur, updates=mis_a_jour, allow_input_downcast=True)
f_get_erreur = theano.function([donnees_entree,labels], erreur, allow_input_downcast=True)
f_get_output = theano.function([donnees_entree],predictions, allow_input_downcast=True)

min_erreur_validation = 1e4

losses_train = []
epochs_train = []
losses_valid = []
epochs_valid = []

for e in range(100):
    loss_train,num_batch=0,0
    for batch in iterate_minibatches(data_train,label_train,100,shuffle=True):
        data,label=batch
        loss_train+=f_train(data,label)
        num_batch+=1
    print("EPOCH#",e,'   ',loss_train/num_batch)
    losses_train.append(loss_train/num_batch)
    epochs_train.append(e)
    if e>0 and e%10==0:
        valid_loss = f_get_erreur(data_val,label_val)
        predictions_val = f_get_output(data_val)
        print("validation loss : ",valid_loss)
        print("proportion guess : ", get_pourcentage_guess(predictions_val,label_val))
        losses_valid.append(valid_loss)
        epochs_valid.append(e)
        if valid_loss<min_erreur_validation :
            parametres_optimaux = lasagne.layers.get_all_param_values(output_layer)
            min_erreur_validation = valid_loss

lasagne.layers.set_all_param_values(output_layer, parametres_optimaux)

print("test: ")
print("\terreur test: ",f_get_erreur(data_test,label_test))
print("\tpourcentage reussi : ",get_pourcentage_guess(f_get_output(data_test),label_test))

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax2 = fig.add_subplot(1,1,1)
ax1.plot(epochs_train,losses_train,color='green', label='training losses')
ax2.plot(epochs_valid,losses_valid, color='red', label='validation losses')

plt.show()
