import warnings
from typing import Optional, Tuple
import math
import torch
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import functional as F
from torch.nn.functional import (pad,_mha_shape_check,_canonical_mask, _in_projection_packed, _in_projection, _none_or_dtype,
                                 softmax, dropout, linear, scaled_dot_product_attention)
from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)

__all__ = ['Threshold', 'ReLU', 'RReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Hardsigmoid', 'Tanh',
           'SiLU', 'Mish', 'Hardswish', 'ELU', 'CELU', 'SELU', 'GLU', 'GELU', 'Hardshrink', 'LeakyReLU',
           'LogSigmoid', 'Softplus', 'Softshrink', 'MultiheadAttention', 'PReLU', 'Softsign', 'Tanhshrink',
           'Softmin', 'Softmax', 'Softmax2d', 'LogSoftmax']



def multi_head_attention_forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Optional[Tensor],
        in_proj_bias: Optional[Tensor],
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        weight_matrix: Optional[Tensor] = None
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        is_causal: If specified, applies a causal mask as attention mask, and ignores
            attn_mask for computing scaled dot product attention.
            Default: ``False``.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True


    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a FloatTensor is provided, it will be directly added to the value.
          If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):  # no
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            is_causal=is_causal,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights,
        )

    is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    key_padding_mask = _canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=_none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype
    )

    if is_causal:
        attn_mask = None

    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:  # yes
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask

    attn_mask = _canonical_mask(
        mask=attn_mask,
        mask_name="attn_mask",
        other_type=_none_or_dtype(key_padding_mask),
        other_name="key_padding_mask",
        target_type=q.dtype,
        check_other=False,
    )

    if attn_mask is not None:
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:  # no
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:  # yes
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:  # yes
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:  # no
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:  # no
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len). \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    if need_weights:  # yes
        B, Nt, E = q.shape
        q_scaled = q / math.sqrt(E)
        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))

        if weight_matrix is not None:
            weight_matrix = weight_matrix.repeat(attn_output_weights.shape[0] // weight_matrix.shape[0], 1, 1)
            attn_output_weights = attn_output_weights + weight_matrix.to(attn_output_weights.device)
            
        attn_output_weights = softmax(attn_output_weights, dim=-1)  # core
        if dropout_p > 0.0:
            attn_output_weights = dropout(attn_output_weights, p=dropout_p)


        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        # attn_mask can be either (L,S) or (N*num_heads, L, S)
        # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
        # in order to match the input for SDPA of (N, num_heads, L, S)
        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, num_heads, src_len, head_dim)

        attn_output = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)

        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None

class Threshold(Module):
    r"""Thresholds each element of the input Tensor.

    Threshold is defined as:

    .. math::
        y =
        \begin{cases}
        x, &\text{ if } x > \text{threshold} \\
        \text{value}, &\text{ otherwise }
        \end{cases}

    Args:
        threshold: The value to threshold at
        value: The value to replace with
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Threshold(0.1, 20)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['threshold', 'value', 'inplace']

    threshold: float
    value: float
    inplace: bool

    def __init__(self, threshold: float, value: float, inplace: bool = False) -> None:
        super().__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace
        # TODO: check in THNN (if inplace == True, then assert value <= threshold)

    def forward(self, input: Tensor) -> Tensor:
        return F.threshold(input, self.threshold, self.value, self.inplace)

    def extra_repr(self):
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'threshold={}, value={}{}'.format(
            self.threshold, self.value, inplace_str
        )


class ReLU(Module):
    r"""Applies the rectified linear unit function element-wise:

    :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/ReLU.png

    Examples::

        >>> m = nn.ReLU()
        >>> input = torch.randn(2)
        >>> output = m(input)


      An implementation of CReLU - https://arxiv.org/abs/1603.05201

        >>> m = nn.ReLU()
        >>> input = torch.randn(2).unsqueeze(0)
        >>> output = torch.cat((m(input), m(-input)))
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class RReLU(Module):
    r"""Applies the randomized leaky rectified liner unit function, element-wise,
    as described in the paper:

    `Empirical Evaluation of Rectified Activations in Convolutional Network`_.

    The function is defined as:

    .. math::
        \text{RReLU}(x) =
        \begin{cases}
            x & \text{if } x \geq 0 \\
            ax & \text{ otherwise }
        \end{cases}

    where :math:`a` is randomly sampled from uniform distribution
    :math:`\mathcal{U}(\text{lower}, \text{upper})`.

     See: https://arxiv.org/pdf/1505.00853.pdf

    Args:
        lower: lower bound of the uniform distribution. Default: :math:`\frac{1}{8}`
        upper: upper bound of the uniform distribution. Default: :math:`\frac{1}{3}`
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/RReLU.png

    Examples::

        >>> m = nn.RReLU(0.1, 0.3)
        >>> input = torch.randn(2)
        >>> output = m(input)

    .. _`Empirical Evaluation of Rectified Activations in Convolutional Network`:
        https://arxiv.org/abs/1505.00853
    """
    __constants__ = ['lower', 'upper', 'inplace']

    lower: float
    upper: float
    inplace: bool

    def __init__(
        self,
        lower: float = 1. / 8,
        upper: float = 1. / 3,
        inplace: bool = False
    ):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.rrelu(input, self.lower, self.upper, self.training, self.inplace)

    def extra_repr(self):
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'lower={}, upper={}{}'.format(self.lower, self.upper, inplace_str)


class Hardtanh(Module):
    r"""Applies the HardTanh function element-wise.

    HardTanh is defined as:

    .. math::
        \text{HardTanh}(x) = \begin{cases}
            \text{max\_val} & \text{ if } x > \text{ max\_val } \\
            \text{min\_val} & \text{ if } x < \text{ min\_val } \\
            x & \text{ otherwise } \\
        \end{cases}

    Args:
        min_val: minimum value of the linear region range. Default: -1
        max_val: maximum value of the linear region range. Default: 1
        inplace: can optionally do the operation in-place. Default: ``False``

    Keyword arguments :attr:`min_value` and :attr:`max_value`
    have been deprecated in favor of :attr:`min_val` and :attr:`max_val`.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Hardtanh.png

    Examples::

        >>> m = nn.Hardtanh(-2, 2)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['min_val', 'max_val', 'inplace']

    min_val: float
    max_val: float
    inplace: bool

    def __init__(
        self,
        min_val: float = -1.,
        max_val: float = 1.,
        inplace: bool = False,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> None:
        super().__init__()
        if min_value is not None:
            warnings.warn("keyword argument min_value is deprecated and rename to min_val")
            min_val = min_value
        if max_value is not None:
            warnings.warn("keyword argument max_value is deprecated and rename to max_val")
            max_val = max_value

        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace
        assert self.max_val > self.min_val

    def forward(self, input: Tensor) -> Tensor:
        return F.hardtanh(input, self.min_val, self.max_val, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'min_val={}, max_val={}{}'.format(
            self.min_val, self.max_val, inplace_str
        )


class ReLU6(Hardtanh):
    r"""Applies the element-wise function:

    .. math::
        \text{ReLU6}(x) = \min(\max(0,x), 6)

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/ReLU6.png

    Examples::

        >>> m = nn.ReLU6()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, inplace: bool = False):
        super().__init__(0., 6., inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class Sigmoid(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}


    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Sigmoid.png

    Examples::

        >>> m = nn.Sigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        return torch.sigmoid(input)


class Hardsigmoid(Module):
    r"""Applies the Hardsigmoid function element-wise.

    Hardsigmoid is defined as:

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            1 & \text{if~} x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Hardsigmoid.png

    Examples::

        >>> m = nn.Hardsigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['inplace']

    inplace: bool

    def __init__(self, inplace : bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.hardsigmoid(input, self.inplace)


class Tanh(Module):
    r"""Applies the Hyperbolic Tangent (Tanh) function element-wise.

    Tanh is defined as:

    .. math::
        \text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Tanh.png

    Examples::

        >>> m = nn.Tanh()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        return torch.tanh(input)

class SiLU(Module):
    r"""Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/SiLU.png

    Examples::

        >>> m = nn.SiLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.silu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class Mish(Module):
    r"""Applies the Mish function, element-wise.
    Mish: A Self Regularized Non-Monotonic Neural Activation Function.

    .. math::
        \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))

    .. note::
        See `Mish: A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Mish.png

    Examples::

        >>> m = nn.Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.mish(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class Hardswish(Module):
    r"""Applies the Hardswish function, element-wise, as described in the paper:
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_.

    Hardswish is defined as:

    .. math::
        \text{Hardswish}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            x & \text{if~} x \ge +3, \\
            x \cdot (x + 3) /6 & \text{otherwise}
        \end{cases}

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Hardswish.png

    Examples::

        >>> m = nn.Hardswish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['inplace']

    inplace: bool

    def __init__(self, inplace : bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.hardswish(input, self.inplace)


class ELU(Module):
    r"""Applies the Exponential Linear Unit (ELU) function, element-wise, as described
    in the paper: `Fast and Accurate Deep Network Learning by Exponential Linear
    Units (ELUs) <https://arxiv.org/abs/1511.07289>`__.

    ELU is defined as:

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}

    Args:
        alpha: the :math:`\alpha` value for the ELU formulation. Default: 1.0
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/ELU.png

    Examples::

        >>> m = nn.ELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['alpha', 'inplace']
    alpha: float
    inplace: bool

    def __init__(self, alpha: float = 1., inplace: bool = False) -> None:
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.elu(input, self.alpha, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'alpha={}{}'.format(self.alpha, inplace_str)


class CELU(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))

    More details can be found in the paper `Continuously Differentiable Exponential Linear Units`_ .

    Args:
        alpha: the :math:`\alpha` value for the CELU formulation. Default: 1.0
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/CELU.png

    Examples::

        >>> m = nn.CELU()
        >>> input = torch.randn(2)
        >>> output = m(input)

    .. _`Continuously Differentiable Exponential Linear Units`:
        https://arxiv.org/abs/1704.07483
    """
    __constants__ = ['alpha', 'inplace']
    alpha: float
    inplace: bool

    def __init__(self, alpha: float = 1., inplace: bool = False) -> None:
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.celu(input, self.alpha, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'alpha={}{}'.format(self.alpha, inplace_str)


class SELU(Module):
    r"""Applied element-wise, as:

    .. math::
        \text{SELU}(x) = \text{scale} * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))

    with :math:`\alpha = 1.6732632423543772848170429916717` and
    :math:`\text{scale} = 1.0507009873554804934193349852946`.

    .. warning::
        When using ``kaiming_normal`` or ``kaiming_normal_`` for initialisation,
        ``nonlinearity='linear'`` should be used instead of ``nonlinearity='selu'``
        in order to get `Self-Normalizing Neural Networks`_.
        See :func:`torch.nn.init.calculate_gain` for more information.

    More details can be found in the paper `Self-Normalizing Neural Networks`_ .

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/SELU.png

    Examples::

        >>> m = nn.SELU()
        >>> input = torch.randn(2)
        >>> output = m(input)

    .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.selu(input, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class GLU(Module):
    r"""Applies the gated linear unit function
    :math:`{GLU}(a, b)= a \otimes \sigma(b)` where :math:`a` is the first half
    of the input matrices and :math:`b` is the second half.

    Args:
        dim (int): the dimension on which to split the input. Default: -1

    Shape:
        - Input: :math:`(\ast_1, N, \ast_2)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(\ast_1, M, \ast_2)` where :math:`M=N/2`

    Examples::

        >>> m = nn.GLU()
        >>> input = torch.randn(4, 2)
        >>> output = m(input)
    """
    __constants__ = ['dim']
    dim: int

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return F.glu(input, self.dim)

    def extra_repr(self) -> str:
        return 'dim={}'.format(self.dim)


class GELU(Module):
    r"""Applies the Gaussian Error Linear Units function:

    .. math:: \text{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    When the approximate argument is 'tanh', Gelu is estimated with:

    .. math:: \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt(2 / \pi) * (x + 0.044715 * x^3)))

    Args:
        approximate (str, optional): the gelu approximation algorithm to use:
            ``'none'`` | ``'tanh'``. Default: ``'none'``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/GELU.png

    Examples::

        >>> m = nn.GELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['approximate']
    approximate: str

    def __init__(self, approximate: str = 'none') -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input, approximate=self.approximate)

    def extra_repr(self) -> str:
        return 'approximate={}'.format(repr(self.approximate))


class Hardshrink(Module):
    r"""Applies the Hard Shrinkage (Hardshrink) function element-wise.

    Hardshrink is defined as:

    .. math::
        \text{HardShrink}(x) =
        \begin{cases}
        x, & \text{ if } x > \lambda \\
        x, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    Args:
        lambd: the :math:`\lambda` value for the Hardshrink formulation. Default: 0.5

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Hardshrink.png

    Examples::

        >>> m = nn.Hardshrink()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['lambd']
    lambd: float

    def __init__(self, lambd: float = 0.5) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, input: Tensor) -> Tensor:
        return F.hardshrink(input, self.lambd)

    def extra_repr(self) -> str:
        return '{}'.format(self.lambd)


class LeakyReLU(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)


    or

    .. math::
        \text{LeakyReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \text{negative\_slope} \times x, & \text{ otherwise }
        \end{cases}

    Args:
        negative_slope: Controls the angle of the negative slope (which is used for
          negative input values). Default: 1e-2
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    .. image:: ../scripts/activation_images/LeakyReLU.png

    Examples::

        >>> m = nn.LeakyReLU(0.1)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['inplace', 'negative_slope']
    inplace: bool
    negative_slope: float

    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.leaky_relu(input, self.negative_slope, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'negative_slope={}{}'.format(self.negative_slope, inplace_str)



class LogSigmoid(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{LogSigmoid}(x) = \log\left(\frac{ 1 }{ 1 + \exp(-x)}\right)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/LogSigmoid.png

    Examples::

        >>> m = nn.LogSigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        return F.logsigmoid(input)


class Softplus(Module):
    r"""Applies the Softplus function :math:`\text{Softplus}(x) = \frac{1}{\beta} *
    \log(1 + \exp(\beta * x))` element-wise.

    SoftPlus is a smooth approximation to the ReLU function and can be used
    to constrain the output of a machine to always be positive.

    For numerical stability the implementation reverts to the linear function
    when :math:`input \times \beta > threshold`.

    Args:
        beta: the :math:`\beta` value for the Softplus formulation. Default: 1
        threshold: values above this revert to a linear function. Default: 20

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Softplus.png

    Examples::

        >>> m = nn.Softplus()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['beta', 'threshold']
    beta: int
    threshold: int

    def __init__(self, beta: int = 1, threshold: int = 20) -> None:
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input: Tensor) -> Tensor:
        return F.softplus(input, self.beta, self.threshold)

    def extra_repr(self) -> str:
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)


class Softshrink(Module):
    r"""Applies the soft shrinkage function elementwise:

    .. math::
        \text{SoftShrinkage}(x) =
        \begin{cases}
        x - \lambda, & \text{ if } x > \lambda \\
        x + \lambda, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    Args:
        lambd: the :math:`\lambda` (must be no less than zero) value for the Softshrink formulation. Default: 0.5

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Softshrink.png

    Examples::

        >>> m = nn.Softshrink()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['lambd']
    lambd: float

    def __init__(self, lambd: float = 0.5) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, input: Tensor) -> Tensor:
        return F.softshrink(input, self.lambd)

    def extra_repr(self) -> str:
        return str(self.lambd)


class MultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    ``forward()`` will use the optimized implementation described in
    `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`_ if all of the following
    conditions are met:

    - self attention is being computed (i.e., ``query``, ``key``, and ``value`` are the same tensor. This
      restriction will be loosened in the future.)
    - inputs are batched (3D) with ``batch_first==True``
    - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor argument ``requires_grad``
    - training is disabled (using ``.eval()``)
    - ``add_bias_kv`` is ``False``
    - ``add_zero_attn`` is ``False``
    - ``batch_first`` is ``True`` and the input is batched
    - ``kdim`` and ``vdim`` are equal to ``embed_dim``
    - if a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ is passed, neither ``key_padding_mask``
      nor ``attn_mask`` is passed
    - autocast is disabled

    If the optimized implementation is in use, a
    `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be passed for
    ``query``/``key``/``value`` to represent padding more efficiently than using a
    padding mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_
    will be returned, and an additional speedup proportional to the fraction of the input
    that is padding can be expected.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> # xdoctest: +SKIP
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135

    """
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super().__setstate__(state)


    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            weight_matrix : Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal : bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and float masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
            If both attn_mask and key_padding_mask are supplied, their types should match.
        is_causal: If specified, applies a causal mask as attention mask. Mutually exclusive with providing attn_mask.
            Default: ``False``.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
          :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
          where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
          embedding dimension ``embed_dim``.
        - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
          returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

        .. note::
            `batch_first` argument is ignored for unbatched inputs.
        """
        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        why_not_fast_path = ''
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
            why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all([(x is None or x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any([x is not None and x.requires_grad for x in tensor_args]):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

                return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    merged_mask,
                    need_weights,
                    average_attn_weights,
                    mask_type)

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
                weight_matrix=weight_matrix)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights




    def merge_masks(self, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                    query: Tensor) -> Tuple[Optional[Tensor], Optional[int]]:
        r"""
        Determine mask type and combine masks if necessary. If only one mask is provided, that mask
        and the corresponding mask type will be returned. If both masks are provided, they will be both
        expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
        and mask type 2 will be returned
        Args:
            attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
            key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
            query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
        Returns:
            merged_mask: merged mask
            mask_type: merged mask type (0, 1, or 2)
        """
        mask_type: Optional[int] = None
        merged_mask: Optional[Tensor] = None

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=F._none_or_dtype(key_padding_mask),
            other_name="key_padding_mask",
            target_type=query.dtype,
            check_other=False,
        )

        if attn_mask is not None:
            mask_type = 0
            merged_mask = attn_mask
        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask
        if (attn_mask is not None) and (key_padding_mask is not None):
            # In this branch query can't be a nested tensor, so it has a shape
            batch_size, seq_len, _ = query.shape
            mask_type = 2
            key_padding_mask_expanded = key_padding_mask.view(batch_size, 1, 1, seq_len) \
                                                        .expand(-1, self.num_heads, -1, -1)
            attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(batch_size, self.num_heads, -1, -1)
            merged_mask = attn_mask_expanded + key_padding_mask_expanded
        return merged_mask, mask_type


class PReLU(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{PReLU}(x) = \max(0,x) + a * \min(0,x)

    or

    .. math::
        \text{PReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        ax, & \text{ otherwise }
        \end{cases}

    Here :math:`a` is a learnable parameter. When called without arguments, `nn.PReLU()` uses a single
    parameter :math:`a` across all input channels. If called with `nn.PReLU(nChannels)`,
    a separate :math:`a` is used for each input channel.


    .. note::
        weight decay should not be used when learning :math:`a` for good performance.

    .. note::
        Channel dim is the 2nd dim of input. When input has dims < 2, then there is
        no channel dim and the number of channels = 1.

    Args:
        num_parameters (int): number of :math:`a` to learn.
            Although it takes an int as input, there is only two values are legitimate:
            1, or the number of channels at input. Default: 1
        init (float): the initial value of :math:`a`. Default: 0.25

    Shape:
        - Input: :math:`( *)` where `*` means, any number of additional
          dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Attributes:
        weight (Tensor): the learnable weights of shape (:attr:`num_parameters`).

    .. image:: ../scripts/activation_images/PReLU.png

    Examples::

        >>> m = nn.PReLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['num_parameters']
    num_parameters: int

    def __init__(self, num_parameters: int = 1, init: float = 0.25,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super().__init__()
        self.weight = Parameter(torch.empty(num_parameters, **factory_kwargs).fill_(init))

    def forward(self, input: Tensor) -> Tensor:
        return F.prelu(input, self.weight)

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)


class Softsign(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{SoftSign}(x) = \frac{x}{ 1 + |x|}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Softsign.png

    Examples::

        >>> m = nn.Softsign()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        return F.softsign(input)


class Tanhshrink(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Tanhshrink}(x) = x - \tanh(x)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Tanhshrink.png

    Examples::

        >>> m = nn.Tanhshrink()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        return F.tanhshrink(input)


class Softmin(Module):
    r"""Applies the Softmin function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range `[0, 1]` and sum to 1.

    Softmin is defined as:

    .. math::
        \text{Softmin}(x_{i}) = \frac{\exp(-x_i)}{\sum_j \exp(-x_j)}

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Args:
        dim (int): A dimension along which Softmin will be computed (so every slice
            along dim will sum to 1).

    Returns:
        a Tensor of the same dimension and shape as the input, with
        values in the range [0, 1]

    Examples::

        >>> m = nn.Softmin(dim=1)
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """
    __constants__ = ['dim']
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input: Tensor) -> Tensor:
        return F.softmin(input, self.dim, _stacklevel=5)

    def extra_repr(self):
        return 'dim={dim}'.format(dim=self.dim)

class Softmax(Module):
    r"""Applies the Softmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range [0,1] and sum to 1.

    Softmax is defined as:

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    When the input Tensor is a sparse tensor then the unspecified
    values are treated as ``-inf``.

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Args:
        dim (int): A dimension along which Softmax will be computed (so every slice
            along dim will sum to 1).

    .. note::
        This module doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use `LogSoftmax` instead (it's faster and has better numerical properties).

    Examples::

        >>> m = nn.Softmax(dim=1)
        >>> input = torch.randn(2, 3)
        >>> output = m(input)

    """
    __constants__ = ['dim']
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input: Tensor) -> Tensor:
        return F.softmax(input, self.dim, _stacklevel=5)

    def extra_repr(self) -> str:
        return 'dim={dim}'.format(dim=self.dim)


class Softmax2d(Module):
    r"""Applies SoftMax over features to each spatial location.

    When given an image of ``Channels x Height x Width``, it will
    apply `Softmax` to each location :math:`(Channels, h_i, w_j)`

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`.
        - Output: :math:`(N, C, H, W)` or :math:`(C, H, W)` (same shape as input)

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Examples::

        >>> m = nn.Softmax2d()
        >>> # you softmax over the 2nd dimension
        >>> input = torch.randn(2, 3, 12, 13)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        assert input.dim() == 4 or input.dim() == 3, 'Softmax2d requires a 3D or 4D tensor as input'
        return F.softmax(input, -3, _stacklevel=5)


class LogSoftmax(Module):
    r"""Applies the :math:`\log(\text{Softmax}(x))` function to an n-dimensional
    input Tensor. The LogSoftmax formulation can be simplified as:

    .. math::
        \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Args:
        dim (int): A dimension along which LogSoftmax will be computed.

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [-inf, 0)

    Examples::

        >>> m = nn.LogSoftmax(dim=1)
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """
    __constants__ = ['dim']
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input: Tensor) -> Tensor:
        return F.log_softmax(input, self.dim, _stacklevel=5)

    def extra_repr(self):
        return 'dim={dim}'.format(dim=self.dim)

