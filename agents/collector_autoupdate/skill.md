# Collector Auto-Update

Analyze framework API changes and update collectors accordingly.

## When to Use

- User says "check collector API changes for trtllm"
- User says "update collectors for new trtllm version"
- User says "analyze API breaking changes"

## Tool

```bash
./tools/api-diff <backend> <old_version> <new_version>

# Example:
./tools/api-diff trtllm v1.2.0rc5 v1.3.0rc3
```

Output (JSON):
```json
{
  "backend": "trtllm",
  "old_version": "v1.2.0rc5",
  "new_version": "v1.3.0rc3",
  "modules": {
    "_torch/attention_backend": {
      "added": 8,
      "removed": 2,
      "changed": 10,
      "breaking_changes": [
        "AttentionBackend methods changed",
        "support_nvfp4_output removed"
      ]
    },
    "_torch/modules/fused_moe": {
      "added": 50,
      "removed": 4,
      "changed": 29
    }
  },
  "summary": {
    "total_added": 227,
    "total_removed": 46,
    "total_changed": 99
  }
}
```

## Workflow

### Step 1: Identify API Changes

```bash
# Clone versions if needed (cached in ~/.cache/aic-schema/git/)
./tools/api-diff trtllm v1.2.0rc5 v1.3.0rc3
```

### Step 2: Analyze Breaking Changes

Focus on:
1. **Removed APIs**: Functions/classes that no longer exist
2. **Changed signatures**: Function parameters changed
3. **Changed class methods**: Methods added/removed/renamed

### Step 3: Update Collectors

For each breaking change:

1. **Check if collector uses the API**:
   ```bash
   grep -r "old_function_name" collector/trtllm/
   ```

2. **If used, create version-specific collector**:
   ```
   collector/trtllm/collect_xxx_v1_3.py
   ```

3. **Or add version detection**:
   ```python
   try:
       from tensorrt_llm._torch.modules.new_api import NewClass
       USE_NEW_API = True
   except ImportError:
       from tensorrt_llm._torch.modules.old_api import OldClass as NewClass
       USE_NEW_API = False
   ```

### Step 4: Test

```bash
# Run collector with new version
python collector/trtllm/collect_xxx.py --test
```

## File Locations

```
aiconfigurator/
├── agents/
│   └── collector_autoupdate/
│       ├── skill.md           # This file
│       └── tools/
│           └── api-diff       # API diff tool
└── collector/
    └── trtllm/
        ├── collect_moe.py     # Current version
        ├── collect_moe_pre_1_0.py  # Old version
        └── collect_moe_v1_3.py     # New version (if needed)
```

## Supported Backends

| Backend | Package | Collector Path |
|---------|---------|----------------|
| trtllm | tensorrt-llm | collector/trtllm/ |
| vllm | vllm | collector/vllm/ |
| sglang | sglang | collector/sglang/ |

## API Modules to Monitor

### trtllm
- `_torch/attention_backend` - Attention implementations
- `_torch/modules/fused_moe` - MoE kernels
- `_torch/modules/mamba` - Mamba2 SSM
- `_torch/pyexecutor` - Executor and KV cache
- `_torch/model_config` - Model configuration

### vllm
- `vllm/engine` - Engine APIs
- `vllm/model_executor` - Model execution

### sglang
- `sglang/srt` - Runtime engine

## Key API Changes v1.2.0rc5 → v1.3.0rc3

### Parameter Changes (Detailed Analysis)

#### ✅ No Changes (Backward Compatible)

| API | Module | Params |
|-----|--------|--------|
| create_attention | _torch/attention_backend/utils | 21 |
| create_moe_backend | _torch/modules/fused_moe | 19 |
| KVCacheParams | _torch/metadata | 5 fields |
| MLAParams | _torch/attention_backend/interface | 8 fields |
| ModelConfig | _torch/model_config | 12 fields |
| causal_conv1d_fn | _torch/modules/mamba | 9 |

#### ⚠️ Changed (Still Backward Compatible)

**selective_state_update** (`_torch/modules/mamba/selective_state_update.py`)
- Added 3 optional parameters (all have defaults):
  - `out=None` - Output buffer
  - `disable_state_update=False` - Skip state update
  - `intermediate_states_buffer=None` - Intermediate states

### Conclusion

**All collector-used APIs are backward compatible.** No need for new collector versions.

### vLLM 0.15.1 Test Results

| Collector | Status | Notes |
|-----------|--------|-------|
| collect_moe.py | ✅ Working | float16 and fp8 both work, 60+ data points |
| collect_attn.py | ✅ Working | Context and generation phases work |
| collect_gemm.py | ❌ Breaking | `RowParallelLinear.input_scale` removed (runtime attribute) |
| collect_mla.py | ❌ Breaking | `get_attn_backend_cls(kv_cache_dtype=...)` parameter removed |

**Performance Data Generated:**
- `moe_perf.txt` (5.8 KB, 60 data points)
- `context_attention_perf.txt` (379 B)
- `generation_attention_perf.txt` (383 B)

**Breaking Changes (vllm v0.14.1 → v0.15.1):**
- `RowParallelLinear.input_scale` attribute removed (affects FP8 quantization)
- `get_attn_backend_cls(kv_cache_dtype=...)` parameter removed
- `which_attn_to_use` function removed (replaced by `get_attn_backend`)
- `MultiHeadLatentAttentionWrapper.forward_cuda` removed
- `MultiHeadLatentAttentionWrapper.forward_native` removed

**Why AST Parsing Missed These:**
- `input_scale` is a runtime attribute (not in class definition)
- `get_attn_backend_cls` parameter change needs deeper analysis

### Structural Changes (Non-Breaking)

### Attention Backend
- Added: `create_output`, `update_helix_param`, DSA sparse APIs
- Removed: `support_nvfp4_output`
- Changed: `AttentionBackend`, `TrtllmAttention` methods

### Fused MoE
- Added: EPLB support, NVLink one-sided communication, DeepEP low latency
- Removed: `_is_using_alltoall`, some backend APIs
- Changed: Communication classes, configurable MoE

### Mamba
- Added: Triton-based causal_conv1d
- Changed: selective_state_update, ssd_combined signatures

### PyExecutor
- Added: HangDetector, new CUDA graph runner, KV cache v2
- Removed: Old request queue methods
- Changed: Executor request queue, CUDA graph runner

## Version Naming Convention

- `collect_xxx.py` - Latest version (works with current release)
- `collect_xxx_pre_1_0.py` - Works before version 1.0
- `collect_xxx_v1_3.py` - Specific version (1.3.x)
- `collect_xxx_1_1rc2.py` - Specific RC version

## Notes

- API changes are detected via AST parsing
- Git cache stores cloned repos in `~/.cache/aic-schema/git/`
- Breaking changes require manual review of collector code

## vLLM 0.15.1 Breaking Changes TODO

### collect_gemm.py
- **Issue**: `RowParallelLinear.input_scale` not found
- **Location**: FP8 quantization path
- **Fix**: Check if `input_scale` exists, create if needed:
  ```python
  if not hasattr(gemm, "input_scale") or gemm.input_scale is None:
      input_scale = torch.ones(1, dtype=torch.float32, device=device)
      gemm.register_parameter("input_scale", torch.nn.Parameter(input_scale))
  ```
- **Action**: Create `collect_gemm_0_15.py` with fix

### collect_mla.py  
- **Issue**: `get_attn_backend_cls(kv_cache_dtype=...)` parameter not accepted
- **Location**: `collector/vllm/collect_mla.py:72-109`
- **Fix**: Use `AttentionSelectorConfig` with `kv_cache_dtype` parameter:
  ```python
  from vllm.v1.attention.selector import AttentionSelectorConfig
  
  attn_selector_config = AttentionSelectorConfig(
      head_size=head_dim,
      dtype=dtype,
      kv_cache_dtype="fp8" if use_fp8_kv_cache else None,
      block_size=block_size,
      use_mla=True,  # MLA mode
  )
  backend = current_platform.get_attn_backend_cls(None, attn_selector_config)
  ```
- **Action**: Create `collect_mla_0_15.py` with fix

### Recommended Workflow
1. Test with `compatible_versions = ["0.15.0", "0.15.1"]`
2. Create version-specific collectors when breaking changes found
3. Update `compatible_versions` list in original collectors

### collect_mla.py Deep Issue

**Problem**: v0.15.1 requires `vllm_config` to be set before calling `get_attn_backend_cls()`

**Root Cause**:
```python
# In vllm/v1/attention/backends/mla/flashinfer_mla.py:81
def supports_combination(...):
    vllm_config = get_current_vllm_config()  # <-- Needs config here!
```

**Current Code Flow**:
1. `run_attention_torch()` starts
2. Calls `get_attn_backend_cls()` (needs config)
3. Later: `set_current_vllm_config(vllm_config)` (too late!)

**Required Fix**: Move `set_current_vllm_config()` earlier, or refactor to avoid `get_attn_backend_cls()` call

**Action**: Create `collect_mla_0_15.py` with refactored flow

### collect_gemm.py FP8 Issue (v0.15.1) - RESOLVED ✅

**Breaking Change**: `Fp8LinearMethod.apply()` now accesses `layer.input_scale` internally

**API Change**:
- **v0.14.1**: `self.fp8_linear.apply(input=..., input_scale=layer.input_scale)`
- **v0.15.1**: `self.fp8_linear.apply_weights(layer, x, bias)` with internal `_get_layer_params()`

**Root Cause**:
1. `create_weights()` doesn't create `input_scale` for dynamic mode
2. `process_weights_after_loading()` would set `layer.input_scale = None`
3. Standalone tests don't call `process_weights_after_loading()`
4. Result: `AttributeError: 'RowParallelLinear' object has no attribute 'input_scale'`

**Fix**:
```python
# v0.15.1+: dynamic mode requires input_scale = None
if not hasattr(gemm, "input_scale"):
    gemm.input_scale = None  # dynamic mode: None is expected
```

**Why this is correct**:
- `_get_layer_params()` uses `getattr(layer, x_s, None)` with default `None`
- `QuantFP8.forward_cuda()` checks `assert (scale is not None) == self.static`
- dynamic mode expects `scale=None`, then computes it from input tensor

**Key Insight**: Dynamic mode quantizes activations at runtime, so `input_scale` should be `None` (not a fake value like `1.0`)
