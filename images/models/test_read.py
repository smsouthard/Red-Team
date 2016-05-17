""
""
import numpy as np
from PIL import Image
import pandas as pd
import theano
import theano.tensor as T
import math

def image_list_read(image_list, mypath):

    driver_list = pd.read_csv(image_list)

    driver_list['path'] = mypath \
                  + driver_list['classname'] \
                  + '/' \
                  + driver_list['img']

    return driver_list


def driver_arrays( driver_list, sample=range(0,1000)):

    image_list = []

    for link in driver_list['path'][sample]:
        image_list.append(np.asarray(Image.open(link).convert('L'),
                                     dtype='float32').flatten())

    return image_list


def label_arrays(driver_list, sample=range(0,1000)):

    labels = []

    for record in driver_list['classname'][sample]:
        labels.append(int(record[1:]))

    return labels


def image_stack_builder(image_list, labels):

    image_list = (np.stack(image_list),
            np.asarray(labels, dtype='float32'))

    return image_list

def sample_batcher(driver_list, batch_length):

    batchlist = []
    n_batches = int(math.ceil(len(driver_list) / batch_length))

    for batch in range(n_batches):
        batchlist.append(driver_list.sample(batch_length))
        driver_list = driver_list.loc[~driver_list.index.isin(batchlist[batch].index)]
        print driver_list.shape + batchlist[batch].shape

     return batchlist

def iteration_machine(batchlist):

    n = range(len(batchlist))

    imagexy = []

    for record in n:
        images = driver_arrays(batchlist[n])
        labels = label_arrays(batchlist[n])
        imagexy.append(image_stack_builder(images, labels))

    return imagexy

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables
       The reason we store our dataset in shared variables is to allow
       Theano to copy it into the GPU memory (when code is run on GPU).
       Since copying data into the GPU is slow, copying a minibatch everytime
       is needed (the default behaviour if the data is not in a shared
       variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

#test_set_x, test_set_y = shared_dataset(test_set)
#valid_set_x, valid_set_y = shared_dataset(valid_set)
#train_set_x, train_set_y = shared_dataset(train_set)

#rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
#        (test_set_x, test_set_y)]
#return rval

def main():

    mypath = '/home/smsouthard/Data/train/'
    image_list = '/home/smsouthard/Data/train/driver_imgs_list.csv'

    driver_list = image_list_read(image_list, mypath)

    image_list =  driver_arrays(driver_list)

    labels = label_arrays(driver_list)

    imagexy = image_stack_builder(image_list, labels)

    data = shared_dataset(imagexy)




if __name__ == "__main__":
    main()



