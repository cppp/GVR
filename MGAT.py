import typing
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Sequential

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value

if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload


class MGATConv(MessagePassing):
    # (Docstrings and argument definitions remain the same as GATConv)
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            add_self_loops: bool = True,
            edge_dim: Optional[int] = None,
            fill_value: Union[float, Tensor, str] = 'mean',
            bias: bool = True,
            residual: bool = False,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual

        # --- ۱. تعریف مسیر ارزش (Value Path) ---
        # این همان self.lin یا ماتریس W اصلی GAT است.
        self.lin = self.lin_src = self.lin_dst = None
        if isinstance(in_channels, int):
            # مسیر Value (برای ترکیب نهایی): W * h
            self.lin = Linear(in_channels, heads * out_channels, bias=False,
                              weight_initializer='glorot')
        # (حالت Bipartite برای سادگی در اینجا حذف شده است)

        # --- ۲. تعریف مسیر توجه (Attention Path) ---
        F_att_inter = 32  # ابعاد میانی برای حدس نوع گره (net0)
        # ******************************************************************************

        if isinstance(in_channels, int):
            # الف) net0: شبکه حدس نوع/نقش گره
            self.net0 = Sequential(
                # Linear(in_channels, F_att_inter, bias=True, weight_initializer='glorot'),
                # # torch.nn.ELU(),
                # torch.nn.ReLU(),
                # Linear(F_att_inter, F_att_inter, bias=False, weight_initializer='glorot'),
                # torch.nn.ReLU(),
                # Linear(F_att_inter, F_att_inter, bias=False, weight_initializer='glorot'),
                # torch.nn.ReLU(),
                Linear(in_channels, F_att_inter, bias=True, weight_initializer='glorot'),
                # torch.nn.LayerNorm(F_att_inter),
                torch.nn.ELU(),
                Linear(F_att_inter, F_att_inter, bias=False, weight_initializer='glorot'),
                # torch.nn.LayerNorm(F_att_inter),
                torch.nn.ELU(),
                Linear(F_att_inter, F_att_inter, bias=False, weight_initializer='glorot'),
                torch.nn.ELU(),

            )
            # ب) W_att: تبدیل نهایی برای توجه (W_att * net0(h))
            self.lin_att = Linear(F_att_inter, heads * out_channels, bias=False,
                                  weight_initializer='glorot')
            # ج) پارامترهای توجه جدید (a_q و a_k)
            self.att_src = Parameter(torch.empty(1, heads, out_channels))
            self.att_dst = Parameter(torch.empty(1, heads, out_channels))
        # (حالت Bipartite برای سادگی در اینجا حذف شده است)

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.empty(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        # The number of output channels:
        total_out_channels = out_channels * (heads if concat else 1)

        if residual:
            self.res = Linear(
                in_channels
                if isinstance(in_channels, int) else in_channels[1],
                total_out_channels,
                bias=False,
                weight_initializer='glorot',
            )
        else:
            self.register_parameter('res', None)

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.lin is not None:
            self.lin.reset_parameters()
        if self.lin_src is not None:
            self.lin_src.reset_parameters()
        if self.lin_dst is not None:
            self.lin_dst.reset_parameters()
        if self.lin_att is not None:
            self.lin_att.reset_parameters()  # ریست کردن W_att جدید
        if self.net0 is not None:
            for layer in self.net0:  # ریست کردن پارامترهای net0
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        if self.res is not None:
            self.res.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    # (متدهای overload برای forward دست نخورده باقی می‌مانند)

    def forward(  # noqa: F811
            self,
            x: Union[Tensor, OptPairTensor],
            edge_index: Adj,
            edge_attr: OptTensor = None,
            size: Size = None,
            return_attention_weights: Optional[bool] = None,
    ) -> Union[
        Tensor,
        Tuple[Tensor, Tuple[Tensor, Tensor]],
        Tuple[Tensor, SparseTensor],
    ]:
        H, C = self.heads, self.out_channels
        res: Optional[Tensor] = None

        # --- ۱. مسیر ارزش (Value Path - برای ترکیب نهایی) ---
        # این همان W * h است. این خروجی نهایی ترکیب شده را تشکیل می‌دهد.
        if isinstance(x, Tensor):
            x_raw = x
            if self.res is not None:
                res = self.res(x_raw)

            # x_value: ویژگی های تبدیل شده برای ترکیب نهایی (V_j)
            x_value_src = x_value_dst = self.lin(x_raw).view(-1, H, C)
        else:  # Bipartite: برای سادگی در اینجا فرض می کنیم رخ نمی دهد
            raise NotImplementedError("Bipartite graphs not implemented in this modified version.")

        # --- ۲. مسیر توجه (Attention Path - برای محاسبه ضرایب آلفا) ---
        # این از net0 برای تولید Query/Key استفاده می‌کند.

        # الف) اعمال net0 برای استخراج نقش/نوع گره
        x_att = self.net0(x_raw)

        # ب) اعمال W_att و تغییر شکل برای Multi-Head
        x_att_src = x_att_dst = self.lin_att(x_att).view(-1, H, C)

        # x در propagate به x_value اشاره خواهد کرد، اما فعلاً آن را به x_value اختصاص می‌دهیم.
        x = (x_value_src, x_value_dst)

        # --- ۳. محاسبه ضرایب توجه بر اساس مسیر توجه ---
        # alpha_src/dst اکنون بر اساس خروجی net0 (x_att_src/dst) محاسبه می شوند.
        alpha_src = (x_att_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_att_dst is None else (x_att_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        # (بخش افزودن Self-Loops بدون تغییر باقی می‌ماند)
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_value_src.size(0)  # استفاده از ابعاد مسیر Value
                if x_value_dst is not None:
                    num_nodes = min(num_nodes, x_value_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "Edge attr and self loops not supported for sparse tensor")

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr,
                                  size=size)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        # مهم: x همچنان مسیر Value (x_value_src/dst) است.
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        # (بقیه کد forward برای تجمیع Multi-Head و Bias بدون تغییر باقی می‌ماند)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if res is not None:
            out = out + res

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    # (متد edge_update بدون تغییر باقی می‌ماند زیرا محاسبات توجه خام را نرمال می‌کند.)
    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    # (متد message بدون تغییر باقی می‌ماند تا از ویژگی‌های W*h_j استفاده کند.)
    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        # x_j در اینجا ویژگی‌های Value (W*h_j) است.
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')