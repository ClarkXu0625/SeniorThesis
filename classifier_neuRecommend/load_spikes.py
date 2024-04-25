# load from labeled waveform dataset (final_dataset.h5)
import tables
import numpy as np


def load_spike():
    h5_path = 'final_dataset.h5'
    # model_save_dir = path_vars['model_save_dir'] # Directory in which trained model will be saved


    # Load equal numbers of waveforms for pos,neg, split into train,test
    # Since positive samples are >> negative, we will subsample from them
    neg_path = '/sorted/neg'
    pos_path = '/sorted/pos'

    neg_waveforms = []
    pos_waveforms = []

    h5 = tables.open_file(h5_path, 'r')
    for x in h5.iter_nodes(neg_path):
        neg_waveforms.append(x[:])

    neg_waveforms = np.concatenate(neg_waveforms, axis=0)

    # pos_waveforms needs to be of length 75, or 750 that can be downsampled
    pos_node_list = list(h5.iter_nodes(pos_path))
    # Waveforms with same length as neg_waveforms
    pos_matched_units = [x for x in pos_node_list
                        if x.shape[1] == neg_waveforms.shape[1]]
    waveforms_per_unit = neg_waveforms.shape[0]//len(pos_matched_units)

    # with tables.open_file(h5_path,'r') as h5:
    for x in pos_matched_units:
        ind = np.min([x.shape[0], waveforms_per_unit])
        pos_waveforms.append(x[:ind, :])
    pos_waveforms = np.concatenate(pos_waveforms, axis=0)
    h5.close()

    neg_label = [0]*neg_waveforms.shape[0]
    pos_label = [1]*pos_waveforms.shape[0]
    fin_labels = np.concatenate([neg_label, pos_label])

    # plot all waveforms
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(zscore(pos_waveforms,axis=-1), interpolation='nearest', aspect = 'auto')
    # ax[0].set_title('True Spikes')
    # ax[0].set_xlabel('Time (AU)')
    # ax[0].set_ylabel('Waveform #')
    # ax[1].imshow(zscore(neg_waveforms,axis=-1), interpolation='nearest', aspect = 'auto')
    # ax[1].set_title('True Noise')
    # ax[1].set_xlabel('Time (AU)')
    # ax[1].set_ylabel('Waveform #')
    # plt.tight_layout()
    # plt.show()

    return [neg_waveforms, pos_waveforms, neg_label, pos_label, fin_labels]

