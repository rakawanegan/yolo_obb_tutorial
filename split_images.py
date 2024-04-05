from ultralytics.data.split_dota import split_trainval, split_test

# split train and val set, with labels.
split_trainval(
    data_root='path/to/DOTAv1.0/',
    save_dir='path/to/DOTAv1.0-split/',
    rates=[0.5, 1.0, 1.5],    # multiscale
    gap=500
)
# split test set, without labels.
split_test(
    data_root='path/to/DOTAv1.0/',
    save_dir='path/to/DOTAv1.0-split/',
    rates=[0.5, 1.0, 1.5],    # multiscale
    gap=500
)
