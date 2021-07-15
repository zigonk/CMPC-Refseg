import CMPC_model
import CMPC_model_origin
import CMPCv2_model
import CMPCv3_model
import CMPCv4_model
import CMPCv5_model
import CMPCv6_model
import CMPC_DGCN_model
import CMPC_DGCN_Att_model
import CMPCv7_model


def get_segmentation_model(name, **kwargs):
    model = eval(name).LSTM_model(**kwargs)
    return model
