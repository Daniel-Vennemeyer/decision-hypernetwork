"""Which kernels Qwen3.5's linear-attention (Gated DeltaNet) layers will actually use.

Builds one tiny layer and inspects it, instead of relying on transformers' module internals,
which change between releases."""


def linear_attention_backend():
    """Returns {'delta_rule': module of the chunked gated-delta-rule kernel,
                'fla': True if that kernel comes from flash-linear-attention,
                'causal_conv1d': True if the causal-conv1d CUDA kernel is used}."""
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5

    config = Qwen3_5TextConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=2,
                               num_key_value_heads=1, intermediate_size=256, vocab_size=100,
                               linear_num_key_heads=2, linear_num_value_heads=2,
                               linear_key_head_dim=32, linear_value_head_dim=32)
    layer = modeling_qwen3_5.Qwen3_5GatedDeltaNet(config, layer_idx=0)
    kernel = getattr(layer, 'chunk_gated_delta_rule', None)
    module = getattr(kernel, '__module__', '') or ''
    try:  # identity check in case a decorator hides the defining module
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_kernel
        is_fla = kernel is fla_kernel or module.startswith('fla')
    except Exception:
        is_fla = False
    return dict(delta_rule=module or 'unknown', fla=is_fla,
                causal_conv1d=getattr(layer, 'causal_conv1d_fn', None) is not None)
