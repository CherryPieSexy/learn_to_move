import torch.nn as nn


afns = {
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'selu': nn.SELU,
    'tanh': nn.Tanh
}


class Layer(nn.Module):
    def __init__(self,
                 in_features, out_features, layer_norm, afn,
                 residual=True, drop=0.0):
        # Layer, Norm, Afn, Drop, Residual
        super().__init__()
        layer = [nn.Linear(in_features, out_features)]
        if layer_norm:
            layer.append(nn.LayerNorm(out_features))
        if afn is not None:
            layer.append(afns[afn]())
        if drop != 0.0:
            layer.append(nn.Dropout(drop))
        self.layer = nn.Sequential(*layer)
        self.residual = residual and in_features == out_features

    def forward(self, layer_in):
        layer_out = self.layer(layer_in)
        if self.residual:
            layer_out = layer_out + layer_in
        return layer_out


class LSTM(nn.Module):
    def __init__(self, in_features, out_features, layer_norm, residual=True):
        # LSTM, Norm, Residual
        super().__init__()
        self.lstm = nn.LSTM(in_features, out_features, batch_first=True)
        if layer_norm:
            self.layer_norm = nn.LayerNorm(out_features)
        else:
            self.layer_norm = None
        self.residual = residual

    def forward(self, x, lstm_state):
        lstm_out, new_lstm_state = self.lstm(x, lstm_state)
        if self.layer_norm is not None:
            lstm_out = self.layer_norm(lstm_out)
        if self.residual:
            lstm_out = lstm_out + x
        return lstm_out, new_lstm_state
