import CMPC_model
import CMPC_model_origin
import CMPCv2_model
import CMPCv3_model
import CMPCv4_model
import CMPCv4_BiLSTM_T2_model
import CMPCv4_BERT_model
import CMPCv5_BiLSTM_model
import CMPCv5_model


def get_segmentation_model(name, **kwargs):
    model = eval(name).LSTM_model(**kwargs)
    return model
