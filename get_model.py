import CMPC_model
import CMPC_model_origin
import CMPCv2_model


def get_segmentation_model(name, **kwargs):
    model = eval(name).LSTM_model(**kwargs)
    return model
