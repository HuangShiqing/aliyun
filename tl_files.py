# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


## Load and save network list npz
def save_npz(save_list=[], name='model.npz', sess=None):
    """Input parameters and the file name, save parameters into .npz file. Use tl.utils.load_npz() to restore.
    Parameters
    ----------
    save_list : a list
        Parameters want to be saved.
    name : a string or None
        The name of the .npz file.
    sess : None or Session
    Examples
    --------
    - Save model to npz
    >>> tl.files.save_npz(network.all_params, name='model.npz', sess=sess)
    - Load model from npz (Method 1)
    >>> load_params = tl.files.load_npz(name='model.npz')
    >>> tl.files.assign_params(sess, load_params, network)
    - Load model from npz (Method 2)
    >>> tl.files.load_and_assign_npz(sess=sess, name='model.npz', network=network)
    Notes
    -----
    If you got session issues, you can change the value.eval() to value.eval(session=sess)
    References
    ----------
    - `Saving dictionary using numpy <http://stackoverflow.com/questions/22315595/saving-dictionary-of-header-information-using-numpy-savez>`_
    """
    ## save params into a list
    save_list_var = []
    if sess:
        save_list_var = sess.run(save_list)
    else:
        try:
            for k, value in enumerate(save_list):
                save_list_var.append(value.eval())
        except:
            print(" Fail to save model, Hint: pass the session into this function, save_npz(network.all_params, name='model.npz', sess=sess)")
    np.savez(name, params=save_list_var)
    save_list_var = None
    del save_list_var
    print("[*] %s saved" % name)

    ## save params into a dictionary
    # rename_dict = {}
    # for k, value in enumerate(save_dict):
    #     rename_dict.update({'param'+str(k) : value.eval()})
    # np.savez(name, **rename_dict)
    # print('Model is saved to: %s' % name)


def load_npz(path='', name='model.npz'):
    """Load the parameters of a Model saved by tl.files.save_npz().
    Parameters
    ----------
    path : a string
        Folder path to .npz file.
    name : a string or None
        The name of the .npz file.
    Returns
    --------
    params : list
        A list of parameters in order.
    Examples
    --------
    - See ``save_npz``
    References
    ----------
    - `Saving dictionary using numpy <http://stackoverflow.com/questions/22315595/saving-dictionary-of-header-information-using-numpy-savez>`_
    """
    ## if save_npz save params into a dictionary
    # d = np.load( path+name )
    # params = []
    # print('Load Model')
    # for key, val in sorted( d.items() ):
    #     params.append(val)
    #     print('Loading %s, %s' % (key, str(val.shape)))
    # return params
    ## if save_npz save params into a list
    d = np.load(path + name)
    # for val in sorted( d.items() ):
    #     params = val
    #     return params
    return d['params']
    # print(d.items()[0][1]['params'])
    # exit()
    # return d.items()[0][1]['params']