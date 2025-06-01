import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    """
    A Simple sequence-to-sequence LSTM that takes in
    (historical + future pump rates) and predicts future ECD.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, horizon):
        super(SimpleLSTM, self).__init__()
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        window_size = x.size(1) - self.horizon
        lstm_out, (h, c) = self.lstm(x)

        horizon_hidden_states = lstm_out[:, window_size:, :]

        out = self.fc(horizon_hidden_states)  
        out = out.squeeze(-1)  
        return out

class Seq2SeqEncoderDecoderLSTM_Augmented(nn.Module):
    """
    A sequence-to-sequence LSTM that accepts separate encoder and decoder inputs.
    The encoder input is augmented with historical ECD values.

    - The encoder LSTM expects inputs of shape:
          [batch_size, window_size, encoder_input_dim]
      where encoder_input_dim = len(input_features) + 1.

    - The decoder LSTM expects inputs of shape:
          [batch_size, horizon, decoder_input_dim]
      where decoder_input_dim = len(input_features).
    """
    def __init__(self, encoder_input_dim, decoder_input_dim, hidden_dim, num_layers, horizon):
        super(Seq2SeqEncoderDecoderLSTM_Augmented, self).__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder_lstm = nn.LSTM(
            input_size=encoder_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.decoder_lstm = nn.LSTM(
            input_size=decoder_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x_enc, x_dec):
        """
        Args:
          x_enc: Tensor of shape [batch_size, window_size, encoder_input_dim]
          x_dec: Tensor of shape [batch_size, horizon, decoder_input_dim]
        Returns:
          out: Tensor of shape [batch_size, horizon] with the predicted future ECD values.
        """
        _, (h_enc, c_enc) = self.encoder_lstm(x_enc)
        dec_out, _ = self.decoder_lstm(x_dec, (h_enc, c_enc))
        out = self.output_layer(dec_out)
        return out.squeeze(-1)            

class LSTMAugmented(nn.Module):
    """
    A simple LSTM that works with the augmented dataset:
    (x_hist, x_dec, y), where:
      x_hist -> [B, window_size, (input_dim + 1)]  # includes historical ECD
      x_dec  -> [B, horizon, input_dim]            # future input features only
    It concatenates x_hist and a padded x_dec into one sequence, and outputs horizon predictions.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, horizon):
        """
        Args:
          input_dim   (int): Number of original input features (WITHOUT the historical ECD).
          hidden_dim  (int): LSTM hidden dimension.
          num_layers  (int): Number of LSTM layers.
          horizon     (int): Number of future steps to predict.
        """
        super(LSTMAugmented, self).__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_size=input_dim + 1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x_hist, x_dec):
        """
        x_hist: [B, window_size, (input_dim + 1)]
        x_dec : [B, horizon, input_dim]  (needs padding to match x_hist's last dim)
        Returns: [B, horizon]
        """
        batch_size = x_hist.size(0)
        window_size = x_hist.size(1)
        horizon = x_dec.size(1)

        zeros = torch.zeros(batch_size, horizon, 1, device=x_dec.device)
        x_dec_padded = torch.cat([x_dec, zeros], dim=-1)
        # => [B, horizon, (input_dim + 1)]

        x_full = torch.cat([x_hist, x_dec_padded], dim=1)
        # => [B, (window_size + horizon), (input_dim + 1)]

        out, (h_n, c_n) = self.lstm(x_full)
        # out shape: [B, (window_size + horizon), hidden_dim]

        out_fut = out[:, -horizon:, :]
        # => [B, horizon, hidden_dim]

        out_fut = self.fc(out_fut)
        # => [B, horizon, 1]

        out_fut = out_fut.squeeze(-1)
        return out_fut


class Seq2SeqEncoderDecoderLSTM(nn.Module):
    """
    A sequence-to-sequence LSTM model that:
      - Encodes the historical window (window_size steps).
      - Decodes the known future pump rates (horizon steps).
      - Outputs the predicted ECD for each of those horizon steps.

    Keeping input_dim, hidden_dim, num_layers, horizon
    """
    def __init__(self, input_dim, hidden_dim, num_layers, horizon):
        super(Seq2SeqEncoderDecoderLSTM, self).__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.decoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x shape = [batch_size, (window_size + horizon), input_dim]
                  e.g., the first 'window_size' rows are historical,
                        the next 'horizon' rows are future pump rates.

        Split x into two parts:
          x_enc = the historical window
          x_dec = the known future pump rates
        """
    
        window_size = x.size(1) - self.horizon 
        x_enc = x[:, :window_size, :]           
        x_dec = x[:, window_size:, :]        

        _, (h_enc, c_enc) = self.encoder_lstm(x_enc)

        dec_out, (h_dec, c_dec) = self.decoder_lstm(x_dec, (h_enc, c_enc))
        # dec_out shape: [batch_size, horizon, hidden_dim]

        out = self.output_layer(dec_out)  # => [batch_size, horizon, 1]
        out = out.squeeze(-1)
        return out


def create_model(model_type: str, input_dim: int, hidden_dim: int, num_layers: int, horizon: int, **kwargs):
    """
    Create a model by name.
    `kwargs` are currently ignored as other architecture options are removed but kept for potential future flexibility.
    """
    if model_type.lower() == 'lstm':
        return SimpleLSTM(input_dim, hidden_dim, num_layers, horizon)
    elif model_type.lower() == 'encdec_lstm':
        return Seq2SeqEncoderDecoderLSTM(input_dim, hidden_dim, num_layers, horizon)
    elif model_type.lower() == 'encdec_lstm_aug':
        encoder_input_dim = input_dim + 1  
        decoder_input_dim = input_dim
        return Seq2SeqEncoderDecoderLSTM_Augmented(encoder_input_dim, decoder_input_dim, hidden_dim, num_layers, horizon)
    elif model_type.lower() == 'lstm_aug':
        return LSTMAugmented(input_dim, hidden_dim, num_layers, horizon)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    