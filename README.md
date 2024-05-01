# Hawkes-LSTM
Official Implementation of HP-LSTM: Hawkes-Process-LSTM based detection of DDos attack for in-vehicle network

## Enverionment Setup

We recommend using Anaconda to manage your environment.

```bash
# Create a new environment
conda create -n hp_lstm python=3.10

# Activate the environment
conda activate hp_lstm

# Install everything
pip install -e .
```

## Examples

You can use `HP_LSTM` as a normal PyTorch module to integrate with your own code.

Check [examples/hp_lstm_inference.py](./examples/hp_lstm_inference.py) for details.
