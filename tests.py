def check_one_hot_encoding_distribution(encoded, not_encoded, binary=False, class_to_train=None):
    if binary:
        print encoded
        print not_encoded
        not_encoded[class_to_train] == encoded['1']
    else:
        print encoded
        print not_encoded
        assert encoded == not_encoded


def check_length_x_y_equal(x,y):
    assert len(x) == len(y)
