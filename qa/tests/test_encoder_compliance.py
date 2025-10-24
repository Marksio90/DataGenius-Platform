
from backend.encoders import safe_one_hot_encoder

def test_safe_one_hot_encoder_sparse_output():
    enc = safe_one_hot_encoder()
    # scikit-learn >=1.5 ma atrybut sparse_output w init, a fitted enc posiada sparse param in attributes
    assert hasattr(enc, 'sparse_output') and enc.sparse_output is True
