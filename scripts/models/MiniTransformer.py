import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        # Creare una matrice di dimensioni [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # 10000.0 default value in math log
        
        # Encoding sinusoidale
        pe[:, 0::2] = torch.sin(position * div_term)  # Posizioni pari
        pe[:, 1::2] = torch.cos(position * div_term)  # Posizioni dispari
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Aggiungere il positional encoding
        x = x + self.pe[:, :x.size(1), :]
        return x

class MiniTransformerPerElement(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        seq_len,
        d_model=128,
        num_heads=16,
        num_layers=1,
        dropout_prob=0.1,
        predict_log_sigma=False,
    ): #layer 1 #0.1 #4
        super(MiniTransformerPerElement, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.predict_log_sigma = bool(predict_log_sigma)
        
        # Layer di embedding
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_len)
        self.dropout_embedding = nn.Dropout(p=dropout_prob)  # Dropout dopo embedding
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout_prob, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classificatore
        self.dropout_classifier = nn.Dropout(p=dropout_prob)  # Dropout prima della classificazione
        self.classifier = nn.Linear(d_model, num_classes)
        self.log_sigma_head = nn.Linear(d_model, 1) if self.predict_log_sigma else None

    def encode(self, x):
        # x: [batch_size, seq_len, input_dim]
        encoder_input = self.embedding(x)  # [batch_size, seq_len, d_model]
        encoder_input = self.positional_encoding(encoder_input)  # [batch_size, seq_len, d_model]
        encoder_input = self.dropout_embedding(encoder_input)  # Applica dropout sull'embedding
        hidden = self.transformer_encoder(encoder_input)  # [batch_size, seq_len, d_model]
        classifier_input = self.dropout_classifier(hidden)  # Applica dropout sull'output del transformer
        return encoder_input, hidden, classifier_input

    def forward(self, x):
        encoder_input, hidden, classifier_input = self.encode(x)
        logits = self.classifier(classifier_input)  # [batch_size, seq_len, num_classes]
        if self.predict_log_sigma:
            log_sigma = self.log_sigma_head(classifier_input)  # [batch_size, seq_len, 1]
            return logits, log_sigma
        return logits

    def forward_with_representations(self, x):
        encoder_input, hidden, classifier_input = self.encode(x)
        outputs = {
            "encoder_input": encoder_input,
            "hidden": hidden,
            "classifier_input": classifier_input,
            "logits": self.classifier(classifier_input),
        }
        if self.predict_log_sigma:
            outputs["log_sigma"] = self.log_sigma_head(classifier_input)
        return outputs
