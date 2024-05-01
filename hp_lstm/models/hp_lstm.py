import torch
import torch.nn as nn


class HP_LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        hks_dim: int = 16,
    ):
        super(HP_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hks_dim = hks_dim

        # Hawkes Process Gate
        self.A_hks = nn.Linear(hks_dim, hidden_size)
        self.B_hks = nn.Linear(hks_dim, hidden_size)
        self.C_hks = nn.Linear(hks_dim, hidden_size)
        self.D_hks = nn.Parameter(torch.zeros(hidden_size))

        # Input Gate
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size)
        self.b_i = nn.Parameter(torch.zeros(hidden_size))

        # Forget Gate
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)
        self.b_f = nn.Parameter(torch.zeros(hidden_size))

        # Cell Gate
        self.W_c = nn.Linear(input_size, hidden_size)
        self.U_c = nn.Linear(hidden_size, hidden_size)
        self.b_c = nn.Parameter(torch.zeros(hidden_size))

        # Output Gate
        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size)
        # @cite: Jozefowicz, Rafal, Wojciech Zaremba, and Ilya Sutskever.
        #        An empirical exploration of recurrent network architectures."
        #        International conference on machine learning. PMLR, 2015.
        self.b_o = nn.Parameter(torch.ones(hidden_size), requires_grad=True)

        # Output Layer
        self.W_y = nn.Linear(hidden_size, output_size)

        self.h = torch.zeros(hidden_size, dtype=torch.float32, device="cuda")
        self.c = torch.zeros(hidden_size, dtype=torch.float32, device="cuda")

    def reset_parameters(self):
        for param in self.parameters():
            if param.data.ndimension() >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.zeros_(param.data)

    def forward(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        theta: torch.Tensor,
        time_span: torch.Tensor,
    ):
        # Shape of alpha: (batch_size, hks_dim)
        # Shape of beta: (batch_size, hks_dim)
        # Shape of theta: (batch_size, hks_dim)

        # Convert time_span from (batch_size,) to (batch_size, 1)
        time_span = time_span.unsqueeze(1).float()

        h_detached = self.h.detach()
        c_detached = self.c.detach()

        # hte
        hawkes_t = torch.tanh(
            self.A_hks(alpha)
            - torch.exp(self.B_hks(beta * time_span))
            + self.C_hks(theta)
        )

        i_t = torch.sigmoid(self.W_i(x) + self.U_i(h_detached) + self.b_i)
        f_t = torch.sigmoid(self.W_f(x) + self.U_f(h_detached) + self.b_f)

        # chm
        c_detached = hawkes_t * (
            f_t * c_detached
            + i_t * torch.tanh(self.W_c(x) + self.U_c(h_detached) + self.b_c)
        )

        o_t = torch.sigmoid(self.W_o(x) + self.U_o(h_detached) + self.b_o)

        h_detached = o_t * torch.tanh(c_detached)

        y_t = self.W_y(h_detached)

        self.h = h_detached.detach()
        self.c = c_detached.detach()

        return y_t, self.h, self.c

