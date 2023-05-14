from tsl.nn.layers.recurrent.base import GraphGRUCellBase

from tsl.nn.layers import GraphConv


class GraphConvGRUCell(GraphGRUCellBase):

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
        forget_gate = graph_convolution_operation(input_size + hidden_size,
                                hidden_size,
                                root_weight=root_weight,
                                bias=bias,
                                cached=cached,
                                **kwargs)
        update_gate = graph_convolution_operation(input_size + hidden_size,
                                hidden_size,
                                root_weight=root_weight,
                                bias=bias,
                                cached=cached,
                                **kwargs)
        candidate_gate = graph_convolution_operation(input_size + hidden_size,
                                   hidden_size,
                                   root_weight=root_weight,
                                   bias=bias,
                                   cached=cached,
                                   **kwargs)

        super(GraphConvGRUCell, self).__init__(
            hidden_size=hidden_size,
            forget_gate=forget_gate,
            update_gate=update_gate,
            candidate_gate=candidate_gate
        )
