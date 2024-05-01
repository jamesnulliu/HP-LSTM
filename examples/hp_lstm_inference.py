import torch

from hp_lstm.models import HP_LSTM

# Configs
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_path = None  # ToDo
batch_size = 32
seq_len = 32
input_size = 8
output_size = 1
hks_dim = 4

# Generate some random data
X = torch.randn(batch_size, seq_len, input_size).to(device)
alpha = torch.randn(batch_size, hks_dim).to(device)
beta = torch.randn(batch_size, hks_dim).to(device)
theta = torch.randn(batch_size, hks_dim).to(device)
time_span = torch.randint(0, seq_len, (batch_size,)).to(device)

# Define the model
# @note You can intergrate the model as a sublayer of your model
model = HP_LSTM(
    input_size=input_size,
    hidden_size=32,
    output_size=output_size,
    hks_dim=hks_dim,
)

# Load the model from a checkpoint
if ckpt_path is not None:
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device)["model_state_dict"]
    )

# Move the model to the device
model.to(device)

# Set the model to evaluation mode
model.eval()

# Forward pass
pred = model(X, alpha, beta, theta, time_span)

# (batch_size, hidden_size, output_size)
print(pred[0].shape)