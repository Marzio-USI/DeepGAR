from tsl.nn.layers.recurrent.base import GraphLSTMCellBase

from tsl.nn.layers import GraphConv


class GraphConvLSTMCell(GraphLSTMCellBase):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 bias: bool = True,
                 root_weight: bool = True,
                 cached: bool = False,
                 activation="relu",
                 graph_convolution_operation=GraphConv,
                 **kwargs):
        self.input_size = input_size
        # instantiate gates
        input_gate = graph_convolution_operation(input_size + hidden_size,
                               hidden_size,
                               bias=bias,
                               root_weight=root_weight,
                               cached=cached,
                               activation=activation,
                               **kwargs)
        forget_gate = graph_convolution_operation(input_size + hidden_size,
                                hidden_size,
                                bias=bias,
                                root_weight=root_weight,
                                cached=cached,
                                activation=activation,
                                **kwargs)
        cell_gate = graph_convolution_operation(input_size + hidden_size,
                              hidden_size,
                              bias=bias,
                              root_weight=root_weight,
                              cached=cached,
                              activation=activation,
                              **kwargs)
        output_gate = graph_convolution_operation(input_size + hidden_size,
                                hidden_size,
                                bias=bias,
                                root_weight=root_weight,
                                cached=cached,
                                activation=activation,
                                **kwargs)

        super(GraphConvLSTMCell, self).__init__(
            hidden_size=hidden_size,
            input_gate=input_gate,
            forget_gate=forget_gate,
            cell_gate=cell_gate,
            output_gate=output_gate,
        )
