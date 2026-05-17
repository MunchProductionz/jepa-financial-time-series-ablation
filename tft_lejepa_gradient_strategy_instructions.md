# Instructions for Updating TFT + LeJEPA Auxiliary Loss Weighting and Gradient Strategy

## 1. Objective

Update the existing TFT + LeJEPA implementation so that the weighting and gradient-flow behavior of the auxiliary LeJEPA arms can be selected from the experiment configuration YAML file.

The model already has LeJEPA auxiliary arms attached after the Transformer blocks of a TFT-like model. The goal of this update is **not** to redesign the auxiliary arms themselves. The goal is to make the way their losses affect the backbone configurable.

The key problem is that auxiliary losses attached to later Transformer blocks are **not layer-local by default**. In standard PyTorch autograd, a loss attached after block `k` affects all trainable parameters upstream of that loss. Therefore, with three Transformer blocks:

```text
B0 → B1 → B2 → B3 → forecast_head → L_prediction
      │    │    │
      │    │    └── L_aux_3
      │    └─────── L_aux_2
      └──────────── L_aux_1
```

standard global backpropagation gives approximately:

```text
B3 receives: L_prediction + L_aux_3
B2 receives: L_prediction + L_aux_2 + L_aux_3
B1 receives: L_prediction + L_aux_1 + L_aux_2 + L_aux_3
B0 receives: L_prediction + L_aux_1 + L_aux_2 + L_aux_3
```

This is important because if the configuration uses larger auxiliary weights for deeper layers, for example:

```text
lambda_aux_1 < lambda_aux_2 < lambda_aux_3
```

then `L_aux_3` may affect the earlier layers more than their own auxiliary losses. This may be undesirable if the purpose of `L_aux_1` is to shape the representation after block 1, `L_aux_2` after block 2, and so on.

The implementation should therefore support multiple strategies for controlling auxiliary loss influence.

---

## 2. Conceptual Background

The main forecasting loss should remain the dominant objective. The LeJEPA losses are auxiliary representation-shaping losses. They should help hidden states become more predictive, stable, non-collapsed, and geometrically well-conditioned, but they should not overwrite the TFT backbone's normal forecasting computation.

The original TFT architecture is built for multi-horizon forecasting and uses recurrent/local temporal processing, self-attention, variable selection, and gating mechanisms. The auxiliary LeJEPA losses should regularize or improve intermediate representations, not break these mechanisms.

LeJEPA-style learning introduces a latent prediction objective and a SIGReg-style regularization objective. In this project, each auxiliary arm is assumed to operate on projected hidden states after a Transformer block.

For example, for layer `k`:

```text
H_k_context = output of Transformer block k for window x[1:T]
H_k_target  = output of Transformer block k for shifted window x[2:T+1]

Z_k_context = projector_k(H_k_context)
Z_k_target  = projector_k(H_k_target)

Z_k_pred = predictor_k(Z_k_context)

L_aux_k = prediction_distance(Z_k_pred, stopgrad_or_not(Z_k_target)) + sigreg_terms
```

The total training loss is conceptually:

```text
L_total = L_prediction + sum_k lambda_aux_k * L_aux_k
```

However, the key implementation question is:

```text
Which backbone parameters should each L_aux_k be allowed to update?
```

---

## 3. Required YAML Configuration Additions

Add a new top-level or model-level config section, for example:

```yaml
lejepa_auxiliary:
  enabled: true

  gradient_strategy:
    name: global_weighted

  target_side:
    detach_target: true
    sigreg_on_target: false

  weights:
    mode: explicit
    values: [0.01, 0.03, 0.10]
    warmup:
      enabled: true
      start_epoch: 0
      end_epoch: 10
      start_scale: 0.0
      end_scale: 1.0

  sigreg:
    alpha: 0.01

  diagnostics:
    log_aux_losses: true
    log_gradient_norms: true
    log_per_layer_gradient_contributions: false
```

The exact names can be adjusted to match the existing project style, but the config should include the following concepts.

### Required fields

```yaml
lejepa_auxiliary.enabled
```

Whether to use the auxiliary losses.

```yaml
lejepa_auxiliary.gradient_strategy.name
```

Which gradient-routing strategy to use.

```yaml
lejepa_auxiliary.weights
```

How to weight the auxiliary losses.

```yaml
lejepa_auxiliary.target_side.detach_target
```

Whether the shifted-window target representation should be detached in the prediction-distance term.

```yaml
lejepa_auxiliary.sigreg.alpha
```

Weight of SIGReg inside each auxiliary loss or globally, depending on the existing implementation.

### Recommended optional fields

```yaml
lejepa_auxiliary.weights.warmup
```

Warm up auxiliary losses so they do not distort randomly initialized representations early in training.

```yaml
lejepa_auxiliary.gradient_strategy.distance_decay_rho
```

Used by distance-decayed gradient routing.

```yaml
lejepa_auxiliary.gradient_strategy.local_recompute
```

Used by local auxiliary-gradient strategies.

```yaml
lejepa_auxiliary.diagnostics.log_gradient_norms
```

Useful for checking whether auxiliary losses dominate the prediction loss.

---

## 4. Strategy 1: Global Weighted Auxiliary Losses

### Name

```yaml
gradient_strategy:
  name: global_weighted
```

### Concept

This is the simplest strategy. Compute all auxiliary losses, multiply them by their configured weights, add them to the prediction loss, and call `loss.backward()` once.

```text
L_total = L_prediction + lambda_1 L_aux_1 + lambda_2 L_aux_2 + lambda_3 L_aux_3
```

This uses standard PyTorch autograd. A loss affects all upstream parameters in its computational graph.

### Effect on three Transformer blocks

```text
B3 receives: L_prediction + lambda_3 L_aux_3
B2 receives: L_prediction + lambda_2 L_aux_2 + lambda_3 L_aux_3
B1 receives: L_prediction + lambda_1 L_aux_1 + lambda_2 L_aux_2 + lambda_3 L_aux_3
B0 receives: L_prediction + lambda_1 L_aux_1 + lambda_2 L_aux_2 + lambda_3 L_aux_3
```

### YAML example

```yaml
lejepa_auxiliary:
  enabled: true
  gradient_strategy:
    name: global_weighted

  weights:
    mode: explicit
    values: [0.01, 0.03, 0.10]
    warmup:
      enabled: true
      start_epoch: 0
      end_epoch: 10
      start_scale: 0.0
      end_scale: 1.0

  target_side:
    detach_target: true
    sigreg_on_target: false
```

### When to use

Use this as the baseline because it is simple and easy to debug.

### Risk

If deeper auxiliary weights are larger, deeper auxiliary losses may dominate earlier layers. This can distort early-layer representations.

### PyTorch implementation notes

Use ordinary loss aggregation:

```python
loss = prediction_loss
for weight, aux_loss in zip(aux_weights, aux_losses):
    loss = loss + weight * aux_loss
loss.backward()
```

Use `tensor.detach()` for the target representation if configured:

```python
target_z = target_z.detach()
```

PyTorch autograd computes gradients by following the computation graph and accumulating gradients into leaf tensors. `Tensor.detach()` returns a tensor detached from the graph, so gradients from that loss term do not flow through the detached tensor.

Relevant PyTorch docs:

- `torch.autograd`: https://docs.pytorch.org/docs/stable/autograd.html
- `torch.autograd.backward`: https://docs.pytorch.org/docs/stable/generated/torch.autograd.backward.html

---

## 5. Strategy 2: Local Auxiliary Losses by Detach + Recompute

### Name

```yaml
gradient_strategy:
  name: local_recompute
```

### Concept

Each auxiliary loss should mostly update only the Transformer block where it is attached, plus its own projection/prediction head.

For three blocks:

```text
L_aux_1 updates B1 + arm1
L_aux_2 updates B2 + arm2
L_aux_3 updates B3 + arm3
L_prediction updates B0 + B1 + B2 + B3 + forecast_head
```

This prevents later auxiliary losses from strongly backpropagating into earlier layers.

### Why this exists

This directly solves the issue where `L_aux_3` influences `B1` more than `L_aux_1` does.

### YAML example

```yaml
lejepa_auxiliary:
  enabled: true
  gradient_strategy:
    name: local_recompute
    detach_lower_inputs: true
    recompute_auxiliary_paths: true

  weights:
    mode: explicit
    values: [0.02, 0.05, 0.10]
    warmup:
      enabled: true
      start_epoch: 0
      end_epoch: 10
      start_scale: 0.0
      end_scale: 1.0

  target_side:
    detach_target: true
    compute_target_with_no_grad: true
    sigreg_on_target: false
```

### Implementation idea

Run the normal forward pass for the prediction loss. Store `H_0`, `H_1`, `H_2`, `H_3`.

Then, for each auxiliary loss, recompute only the local block using a detached input:

```text
For L_aux_1:
  H_1_aux = B1(detach(H_0))

For L_aux_2:
  H_2_aux = B2(detach(H_1))

For L_aux_3:
  H_3_aux = B3(detach(H_2))
```

Then apply the corresponding auxiliary arm to the local auxiliary representation.

### Effect

The detached input prevents the auxiliary loss from flowing into earlier blocks.

For example:

```text
H_2_aux = B2(H_1.detach())
```

means `L_aux_2` can update `B2`, but not `B1` or `B0`.

### PyTorch implementation notes

Use `detach()` on the lower-layer input to the local auxiliary recomputation.

Use `torch.no_grad()` for the target branch if target representations should act as stable targets:

```python
with torch.no_grad():
    target_hidden_states = model.encode_backbone(target_window)
```

Be careful if Transformer blocks contain dropout. Recomputing a block can produce a different stochastic output from the main pass. Options:

1. accept this as additional regularization;
2. control RNG state carefully;
3. disable dropout in auxiliary recomputation;
4. reuse main hidden states for target values but detach them.

Prefer this strategy if clean layer-local auxiliary shaping is more important than compute efficiency.

### Tradeoff

This strategy is safer but less globally coordinated. `L_aux_3` will no longer help reshape `B1` and `B2` so that `B3` receives better inputs. The main prediction loss still globally coordinates the model.

---

## 6. Strategy 3: Distance-Decayed Auxiliary Gradient Routing

### Name

```yaml
gradient_strategy:
  name: distance_decayed
```

### Concept

Each auxiliary loss should strongly affect its own layer, weakly affect the layer below, and barely affect much earlier layers.

For auxiliary loss `k` affecting block `j`, where `j <= k`, use:

```text
scale(k, j) = rho ** (k - j)
```

Example with `rho = 0.25`:

```text
L_aux_1 → B1: 1.0000

L_aux_2 → B2: 1.0000
L_aux_2 → B1: 0.2500

L_aux_3 → B3: 1.0000
L_aux_3 → B2: 0.2500
L_aux_3 → B1: 0.0625
```

This gives a compromise between local shaping and global coordination.

### YAML example

```yaml
lejepa_auxiliary:
  enabled: true
  gradient_strategy:
    name: distance_decayed
    rho: 0.25
    implementation: manual_autograd_grad

  weights:
    mode: explicit
    values: [0.02, 0.05, 0.10]
    warmup:
      enabled: true
      start_epoch: 0
      end_epoch: 10
      start_scale: 0.0
      end_scale: 1.0

  target_side:
    detach_target: true
    sigreg_on_target: false
```

### Effect on gradients

With three blocks:

```text
B1 receives:
  L_prediction
  + lambda_1 * L_aux_1
  + rho * lambda_2 * L_aux_2
  + rho^2 * lambda_3 * L_aux_3

B2 receives:
  L_prediction
  + lambda_2 * L_aux_2
  + rho * lambda_3 * L_aux_3

B3 receives:
  L_prediction
  + lambda_3 * L_aux_3
```

This is often the most principled strategy for your problem.

### PyTorch implementation notes

PyTorch does not provide a simple built-in option that says: “backpropagate this loss to each earlier layer with a different scale.” Standard `.backward()` sums gradients according to the graph.

To implement this strategy, use `torch.autograd.grad` to compute gradients from each auxiliary loss with respect to selected parameter groups, then manually scale and add those gradients to each parameter’s `.grad`.

Recommended approach:

1. Run the forward pass and compute `prediction_loss`, `aux_loss_1`, `aux_loss_2`, `aux_loss_3`.
2. Call `prediction_loss.backward(retain_graph=True)` to populate normal supervised gradients.
3. For each auxiliary loss, call `torch.autograd.grad` on selected parameter groups.
4. Multiply returned gradients by the desired routing scale.
5. Add them manually into `param.grad`.
6. Call optional gradient clipping.
7. Call `optimizer.step()`.

Useful PyTorch API:

- `torch.autograd.grad`: https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html
- `torch.autograd.backward`: https://docs.pytorch.org/docs/stable/generated/torch.autograd.backward.html
- `Tensor.grad`: populated for leaf tensors after backward/gradient computation.

### Implementation cautions

Be careful with:

```text
retain_graph=True
mixed precision / GradScaler
DDP gradient synchronization
gradient clipping
gradient accumulation across mini-batches
optimizer.zero_grad(set_to_none=True)
allow_unused=True for parameter groups that may not participate in a given loss
```

For a first version, implement this without mixed precision and without DDP. Add those back once the gradient routing is verified.

---

## 7. Strategy 4: Gradient-Scaled Hidden States

### Name

```yaml
gradient_strategy:
  name: hidden_gradient_scaling
```

### Concept

Instead of manually computing parameter gradients, insert custom autograd operations that pass the forward value unchanged but scale the backward gradient.

This is conceptually similar to gradient reversal layers, except the scale is positive and usually less than 1.

Example:

```text
forward:  y = x
backward: dL/dx = scale * dL/dy
```

### YAML example

```yaml
lejepa_auxiliary:
  enabled: true
  gradient_strategy:
    name: hidden_gradient_scaling
    scales:
      aux_1_to_lower: 1.0
      aux_2_to_lower: 0.25
      aux_3_to_lower: 0.0625

  weights:
    mode: explicit
    values: [0.02, 0.05, 0.10]
```

### PyTorch implementation notes

Implement a custom `torch.autograd.Function`:

```python
class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None
```

Then wrap hidden states before feeding them into auxiliary recomputation paths.

This strategy is less precise than manual `torch.autograd.grad` if you need exact per-loss/per-layer routing, but it can be easier to integrate than manual gradient accumulation.

Relevant PyTorch docs:

- Custom autograd functions: https://docs.pytorch.org/docs/stable/autograd.html
- Tensor hooks and autograd internals are also possible, but custom functions are usually clearer for explicit gradient scaling.

### Caution

A simple gradient-scaling wrapper on `H_k` only scales the gradient at that tensor. If the same hidden tensor also receives gradients from multiple losses, gradients may already be summed depending on where the hook/function is inserted. For precise per-loss routing, prefer detach + recompute or manual `torch.autograd.grad`.

---

## 8. Strategy 5: Equal or Normalized Auxiliary Weights with Global Backpropagation

### Name

```yaml
gradient_strategy:
  name: global_normalized
```

### Concept

Use ordinary global backpropagation, but avoid making later auxiliary losses much larger. This reduces the chance that `L_aux_3` dominates early layers.

Possible weight modes:

```yaml
weights:
  mode: equal
  base_value: 0.03
```

or:

```yaml
weights:
  mode: inverse_depth
  values: [0.06, 0.03, 0.015]
```

or:

```yaml
weights:
  mode: normalized_by_gradient_norm
  target_aux_to_prediction_ratio: 0.2
```

### Use case

This is useful as a simple baseline if local or manual gradient routing is too much engineering effort.

### Implementation notes

For `equal` or `explicit`, implementation is simple.

For `normalized_by_gradient_norm`, estimate gradient norms from each loss with respect to relevant parameter groups using `torch.autograd.grad`. Use those norms to rescale auxiliary losses so they do not exceed a configured fraction of the prediction gradient.

This is more expensive, but it provides valuable diagnostics.

---

## 9. Recommended Test Matrix

Add config presets for at least these experiments.

### Experiment A: baseline without LeJEPA

```yaml
lejepa_auxiliary:
  enabled: false
```

Purpose: verify TFT baseline.

---

### Experiment B: global weighted, increasing by depth

```yaml
lejepa_auxiliary:
  enabled: true
  gradient_strategy:
    name: global_weighted
  weights:
    mode: explicit
    values: [0.01, 0.03, 0.10]
    warmup:
      enabled: true
      start_epoch: 0
      end_epoch: 10
      start_scale: 0.0
      end_scale: 1.0
  target_side:
    detach_target: true
```

Purpose: test the original intuitive strategy.

Expected risk: later auxiliary losses may dominate earlier layers.

---

### Experiment C: global weighted, equal weights

```yaml
lejepa_auxiliary:
  enabled: true
  gradient_strategy:
    name: global_weighted
  weights:
    mode: explicit
    values: [0.03, 0.03, 0.03]
    warmup:
      enabled: true
      start_epoch: 0
      end_epoch: 10
      start_scale: 0.0
      end_scale: 1.0
  target_side:
    detach_target: true
```

Purpose: test whether depth-increasing weights are actually necessary.

---

### Experiment D: local recompute, increasing by depth

```yaml
lejepa_auxiliary:
  enabled: true
  gradient_strategy:
    name: local_recompute
    detach_lower_inputs: true
    recompute_auxiliary_paths: true
  weights:
    mode: explicit
    values: [0.02, 0.05, 0.10]
    warmup:
      enabled: true
      start_epoch: 0
      end_epoch: 10
      start_scale: 0.0
      end_scale: 1.0
  target_side:
    detach_target: true
    compute_target_with_no_grad: true
```

Purpose: test layer-local representation shaping.

---

### Experiment E: distance-decayed gradients

```yaml
lejepa_auxiliary:
  enabled: true
  gradient_strategy:
    name: distance_decayed
    rho: 0.25
    implementation: manual_autograd_grad
  weights:
    mode: explicit
    values: [0.02, 0.05, 0.10]
    warmup:
      enabled: true
      start_epoch: 0
      end_epoch: 10
      start_scale: 0.0
      end_scale: 1.0
  target_side:
    detach_target: true
```

Purpose: test the preferred compromise: local dominance with weak cross-layer influence.

---

## 10. Target-Side Detachment Recommendation

Default to:

```yaml
target_side:
  detach_target: true
```

This means the shifted-window latent target is treated as a fixed target for the prediction-distance part of the auxiliary loss.

Use:

```python
loss_aux_k = distance(predicted_z_context, z_target.detach())
```

Conceptually:

```text
Good default:
  Change the context representation so it predicts the shifted target representation.

Riskier alternative:
  Let both context and target representations move toward each other.
```

If SIGReg is applied to target representations, make this independently configurable:

```yaml
target_side:
  detach_target: true
  sigreg_on_target: false
```

Recommended initial setting:

```text
detach target in prediction-distance term
apply SIGReg only to context-side projected embeddings
```

This is conservative and reduces the risk of unstable target chasing.

---

## 11. Warmup Recommendation

Auxiliary losses should usually be warmed up.

Add support for:

```yaml
weights:
  warmup:
    enabled: true
    start_epoch: 0
    end_epoch: 10
    start_scale: 0.0
    end_scale: 1.0
```

The effective auxiliary weight should be:

```text
effective_lambda_k = base_lambda_k * warmup_scale(current_step_or_epoch)
```

Use a linear warmup initially. Optionally add cosine or sigmoid warmup later.

Rationale:

```text
Early training: let TFT learn basic forecasting structure.
Later training: let LeJEPA shape internal representation geometry.
```

---

## 12. Diagnostics and Safety Checks

The implementation should log:

```text
prediction loss
auxiliary loss per layer
effective auxiliary weight per layer
weighted auxiliary loss per layer
total auxiliary loss
ratio: weighted_aux_total / prediction_loss
```

If feasible, also log gradient norms:

```text
||grad_prediction(block_k)||
||grad_aux_1(block_k)||
||grad_aux_2(block_k)||
||grad_aux_3(block_k)||
```

Useful ratios:

```text
aux_to_prediction_grad_ratio_block_1
aux_to_prediction_grad_ratio_block_2
aux_to_prediction_grad_ratio_block_3
```

Recommended initial heuristic:

```text
Early layers:
  total auxiliary gradient should usually be smaller than prediction gradient.

Later layers:
  auxiliary gradient can be closer to prediction gradient, but should not overwhelm it.
```

If `L_aux_3` dominates `B1`, consider:

```text
use local_recompute
use distance_decayed with smaller rho
reduce lambda_3
increase warmup duration
use gradient clipping
```

---

## 13. Implementation Checklist

The agent should complete the following.

### Config parsing

- Add `lejepa_auxiliary.gradient_strategy.name`.
- Add auxiliary weight configuration.
- Add warmup configuration.
- Add target-side detachment configuration.
- Add optional distance-decay parameter `rho`.
- Add diagnostics flags.

### Training loop

- Compute prediction loss as before.
- Compute auxiliary losses according to existing auxiliary-arm implementation.
- Apply the selected gradient strategy.
- Ensure inference path is unchanged and auxiliary arms are not used for prediction.

### Supported strategies

Implement at least:

```text
global_weighted
local_recompute
```

Preferably also implement:

```text
distance_decayed
```

### Validation

- Confirm `lejepa_auxiliary.enabled: false` reproduces the original TFT training behavior.
- Confirm `global_weighted` behaves like a normal summed loss.
- Confirm `local_recompute` prevents `L_aux_3` from updating `B1` and `B2` except through the main prediction loss.
- Confirm `distance_decayed` applies the intended gradient scales.

### Debugging tools

Use:

```python
param.grad
module.parameters()
torch.autograd.grad(...)
tensor.detach()
torch.no_grad()
tensor.retain_grad()
tensor.register_hook(...)
```

`retain_grad()` can be useful for inspecting intermediate hidden-state gradients. `register_hook()` can be useful for debugging or simple gradient modification, but it is less suitable for precise per-loss/per-layer routing once gradients from multiple paths have already been summed.

Relevant PyTorch docs:

- Autograd overview: https://docs.pytorch.org/docs/stable/autograd.html
- `torch.autograd.grad`: https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html
- `torch.autograd.backward`: https://docs.pytorch.org/docs/stable/generated/torch.autograd.backward.html
- Leaf vs non-leaf tensors and `retain_grad`: https://docs.pytorch.org/tutorials/beginner/understanding_leaf_vs_nonleaf_tutorial.html

---

## 14. Recommended Final Design

The most defensible design is:

```text
Primary forecasting loss:
  updates the full TFT backbone normally.

Auxiliary losses:
  update their own arms and shape backbone representations.

Default target behavior:
  detach target-side projected representation in the prediction-distance term.

Initial strategy:
  start with global_weighted and small/equal weights to establish baseline.

Preferred strategy for serious experiments:
  use distance_decayed or local_recompute.
```

If engineering time is limited, implement this order:

```text
1. global_weighted
2. local_recompute
3. distance_decayed
```

The key design principle is:

```text
Do not confuse auxiliary-loss scalar weights with actual per-layer influence.
Actual influence depends on both the loss weight and the gradient path.
```

Therefore, the most useful configuration is one that controls both:

```text
lambda_k: how important auxiliary loss k is
s_kj: how much auxiliary loss k is allowed to influence block j
```

For three Transformer blocks, the preferred conceptual routing is:

```text
B1:
  L_prediction + L_aux_1 + weak L_aux_2 + very weak L_aux_3

B2:
  L_prediction + L_aux_2 + weak L_aux_3

B3:
  L_prediction + L_aux_3
```

This gives each layer a representation-shaping signal from its own auxiliary arm while still allowing deeper objectives to weakly coordinate lower-level representations.
