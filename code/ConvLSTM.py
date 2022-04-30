import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, h_channels, kernel_size, device):
        super(ConvLSTMCell, self).__init__()

        self.h_channels = h_channels
        padding = kernel_size // 2, kernel_size // 2
        self.conv = nn.Conv2d(in_channels=in_channels + h_channels,
                              out_channels=4 * h_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=True)
        self.device = device

    def forward(self, input_data, prev_state):
        h_prev, c_prev = prev_state
        combined = torch.cat((input_data, h_prev), dim=1)  # concatenate along channel axis

        combined_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_output, self.h_channels, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)

        return h_cur, c_cur

    def init_hidden(self, batch_size, image_size):
        """ initialize the first hidden state as zeros """
        height, width, dimensions = image_size
        return (torch.zeros(batch_size, self.h_channels, height, width, dimensions),
                torch.zeros(batch_size, self.h_channels, height, width, dimensions))
        return (torch.zeros(batch_size, self.h_channels, height, width, dimensions,device=self.device),
                torch.zeros(batch_size, self.h_channels, height, width, dimensions,device=self.device))



class ConvLSTM(nn.Module):

    def __init__(
            self,
            in_channels,
            h_channels,
            num_layers,
            kernel_size,
            device):

        super(ConvLSTM, self).__init__()

        self.in_channels = in_channels
        self.num_layer = num_layers
        self.count = 0
        #self.hidden_state_list = []
        #self.cell_state_list = []

        layer_list = []
        for i in range(num_layers):
            cur_in_channels = in_channels if i == 0 else h_channels[i - 1]
            layer_list.append(ConvLSTMCell(in_channels=cur_in_channels,
                                           h_channels=h_channels[i],
                                           kernel_size=kernel_size,
                                           device=device))

        self.layer_list = nn.ModuleList(layer_list)

    def forward(self, x, states=None):
        self.count+=1
        print(self.count)
        print(type(states))
        print(states[0],states[1])
        image_size = (x.size(dim=2),x.size(dim=3),x.size(dim=1))
        batch_size = (x.size(dim=0))
        #my_result = all(elem is None for elem in states)
        if states[0] is None and states[1] is None:
            hidden_states, cell_states = self.init_hidden(batch_size, image_size)
        else:
            hidden_states, cell_states = states
        for i, layer in enumerate(self.layer_list):
            if(i==0):
                h_cur, c_cur = layer(x,(hidden_states[i], cell_states[i]))
                hidden_states[i] = h_cur
                cell_states[i] = c_cur
            else:
                h_cur, c_cur = layer(hidden_states[i-1],(hidden_states[i], cell_states[i]))
                hidden_states[i] = h_cur
                cell_states[i] = c_cur
        # [TODO: layer forward] (Done, I reckon)
        return hidden_states, (hidden_states, cell_states)

    def init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layer):
            init_states.append(self.layer_list[i].init_hidden(batch_size,image_size))
        return init_states


# [TODO: initialze hidden and cell states] (Done)


def activation_factory(name):
    """
    Returns the activation layer corresponding to the input activation name.
    Parameters
    ----------
    name : str
        'relu', 'leaky_relu', 'elu', 'sigmoid', or 'tanh'. Adds the corresponding activation function after the
        convolution.
    """
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'leaky_relu':
        return nn.LeakyReLU(0.2, inplace=True)
    if name == 'elu':
        return nn.ELU(inplace=True)
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'tanh':
        return nn.Tanh()
    if name is None or name == "identity":
        return nn.Identity()

    raise ValueError(f'Activation function `{name}` not yet implemented')


def make_conv_block(conv):
    out_channels = conv.out_channels
    modules = [conv]
    modules.append(nn.GroupNorm(16, out_channels))
    modules.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*modules)


class DCGAN64Encoder(nn.Module):

    def __init__(self, in_c, out_c):
        super(DCGAN64Encoder, self).__init__()

        h_c = out_c // 2
        self.conv = nn.ModuleList([
            make_conv_block(nn.Conv2d(in_c, h_c, 3, 2, 1)),
            make_conv_block(nn.Conv2d(h_c, h_c, 3, 1, 1)),
            make_conv_block(nn.Conv2d(h_c, out_c, 3, 2, 1))])

    def forward(self, x):
        out = x
        for layer in self.conv:
            out = layer(out)
        return out


class DCGAN64Decoder(nn.Module):

    def __init__(self, in_c, out_c, last_activation=None):
        super(DCGAN64Decoder, self).__init__()

        h_c = in_c // 2
        self.conv = nn.ModuleList([
            make_conv_block(nn.ConvTranspose2d(in_c, h_c, 3, 2, 1, output_padding=1)),
            make_conv_block(nn.ConvTranspose2d(h_c, h_c, 3, 1, 1)),
            make_conv_block(nn.ConvTranspose2d(h_c, out_c, 3, 2, 1, output_padding=1))])

        self.last_activation = activation_factory(last_activation)

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.conv):
            out = layer(out)
        return self.last_activation(out)


class Seq2Seq(nn.Module):

    def __init__(self, args):

        super(Seq2Seq, self).__init__()

        self.h_channels = args.h_channels
        self.num_layers = args.num_layers
        self.kernel_size = args.kernel_size
        self.seq_len = args.seq_len
        self.horizon = args.horizon
        self.in_channels = args.in_channels
        self.gpu = args.gpu_id
        self.out_channels = self.in_channels
        self.frame_encoder = DCGAN64Encoder(self.in_channels, self.h_channels).to(self.gpu)
        self.frame_decoder = DCGAN64Decoder(self.h_channels, self.out_channels).to(self.gpu)

        self.model = ConvLSTM(in_channels=self.h_channels,
                              h_channels=[self.h_channels] * self.num_layers,
                              num_layers=self.num_layers,
                              kernel_size=self.kernel_size,
                              device=self.gpu)

    def forward(self, in_seq, out_seq, teacher_forcing_rate=None):

        next_frames = []
        hidden_states, states = None, None
        # encoder
        for t in range(self.seq_len - 1):
            x = self.frame_encoder(in_seq[t])
            hidden_states, states = self.model(x,(hidden_states, states))

        # [TODO: call ConvLSTM]
        # decoder
        for t in range(self.horizon):

            if teacher_forcing_rate is None:
                continue
            # [TODO: use predicted frames as the input]
            else:
                continue
            # [TODO: choose from predicted frames and out_seq as the input
            # [      based on teacher forcing rate]

            # [TODO: call ConvLSTM]
            hidden_states, states = self.model(hidden_states, (hidden_states, states))
            out = self.frame_decoder(hidden_states[-1])
            next_frames.append(out)

        next_frames = torch.stack(next_frames, dim=1)
        return next_frames