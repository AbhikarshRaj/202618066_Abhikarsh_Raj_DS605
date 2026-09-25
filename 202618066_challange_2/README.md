# SGDRegressor Hyperparameter Optimization with Minimum Oracle Calls

## Project Overview

The objective of this task was to optimize the hyperparameters of an **SGDRegressor** model using an Oracle API.

The main challenge was that every model evaluation required an API call, and excessive API calls attracted a penalty. Therefore, the objective was not only to minimize the model loss, but also to achieve a strong result using as few Oracle evaluations as possible.

The Oracle provided the following hyperparameters:

- `eta0`
- `loss`
- `alpha`
- `average`
- `penalty`
- `learning_rate`

The complete search space contained:

**2304 possible hyperparameter configurations**

Testing all 2304 combinations through Grid Search would have required too many API calls. Therefore, a more efficient search strategy was required.

---

## Main Challenge

Normally, hyperparameter optimization can be performed using libraries and techniques such as:

- Grid Search
- Random Search
- Bayesian Optimization
- Optuna
- Scikit-learn tuning utilities

However, external optimization libraries were not allowed in this task.

This meant that the search had to be performed manually while carefully deciding which configuration should be evaluated next.

Therefore, the problem became:

> Find a very low-loss SGDRegressor configuration while minimizing the number of Oracle API calls.

---

## Search Strategy

The optimization started from a baseline configuration.

The initial baseline produced a loss of:

**Baseline Loss = 7.7731686227**

This result was treated as the reference point for subsequent experiments.

Instead of randomly testing configurations, the search followed an iterative strategy.

The basic approach was:

**Current Best Configuration → Modify Parameter → Evaluate → Compare Loss → Keep Improvement**

If a candidate configuration produced a lower loss, it became the new reference configuration.

If it performed worse, the change was rejected.

---

## Initial Learning-Rate Exploration

One of the first parameters explored was `eta0`, which controls the initial learning rate used by SGD.

The baseline used:

**eta0 = 0.01**

with loss:

**7.7731686227**

Changing `eta0` to:

**0.001**

improved the loss to:

**7.7389669126**

This was an improvement of approximately:

**0.03420**

Therefore, the smaller learning rate was initially accepted as a better choice.

Another smaller value was then tested:

**eta0 = 0.0001**

This resulted in:

**Loss = 48.5448701129**

which was dramatically worse.

This experiment was useful because it showed that making the learning rate extremely small did not improve the model.

The search could therefore avoid spending additional calls exploring that region.

---

## Testing Parameter Averaging

The `average` parameter was also tested.

When averaging was enabled, the loss became:

**12.4151212076**

This was significantly worse than the current best loss of approximately:

**7.739**

Therefore:

**average = True**

was rejected, and the search continued with:

**average = False**

This is an example of how a single API call helped eliminate an unpromising option.

---

## Exploring Regularization

The regularization parameters `alpha` and `penalty` were also investigated.

After testing combinations of regularization settings, a promising configuration was found using:

- **alpha = 0.000001**
- **penalty = l1**

This produced:

**Loss = 7.7387602539**

This was a small but measurable improvement over the earlier result:

**7.7389669126**

At this stage, the best observed loss had decreased from:

**7.7731686227 → 7.7387602539**

---

## Moving Beyond Simple One-Parameter Search

Initially, one parameter was changed at a time.

This approach was useful because it made it easy to understand the effect of individual hyperparameters.

However, SGDRegressor hyperparameters can interact.

For example:

- `eta0` interacts with the learning-rate strategy.
- `alpha` affects regularization and can also interact with optimization behavior.
- The best learning rate under one penalty may not be the best learning rate under another penalty.

Because of these interactions, the search was later extended to test promising parameter combinations.

This allowed the optimization to escape configurations that were locally good but not necessarily the best available.

---

## Pairwise and Joint Parameter Exploration

After the initial search, a stronger configuration with loss around:

**7.73186015**

was used as the new anchor.

From this configuration, selected parameter changes were explored.

One important improvement occurred when:

**eta0 = 0.05**

was tested.

The resulting loss was:

**7.7045678683**

This was an important discovery.

Earlier experiments might have suggested that smaller learning rates were preferable, but after other parameters had changed, the larger learning rate became beneficial.

This demonstrated why hyperparameter interaction is important.

A parameter that performs poorly under one configuration may perform much better after other parameters have been optimized.

---

## Learning-Rate Strategy Improvement

The learning-rate schedule was then explored.

Changing the learning-rate strategy to:

**learning_rate = optimal**

produced the strongest result observed during the search:

**Loss = 7.6139588414**

This was a significant improvement over the previous best:

**7.7045678683**

The improvement was approximately:

**0.09061**

A subsequent evaluation confirmed that the loss remained:

**7.6139588414**

indicating that the current configuration was stable under the Oracle evaluation.

---

## Optimization Progress

The main improvement stages were:

| Stage | Loss |
|---|---:|
| Initial Baseline | **7.7731686227** |
| `eta0 = 0.001` | **7.7389669126** |
| Regularization improvement | **7.7387602539** |
| Improved anchor configuration | **7.73186015** |
| `eta0 = 0.05` | **7.7045678683** |
| `learning_rate = optimal` | **7.6139588414** |

The overall loss reduction from the initial baseline to the best configuration was:

**7.7731686227 − 7.6139588414 = 0.1592097813**

Therefore, the search improved the Oracle loss by approximately:

**0.15921**

without exhaustively evaluating all 2304 configurations.

---

## API Call Efficiency

The initial baseline required:

**1 Oracle call**

The first learning-rate improvements occurred within only a few calls.

Further exploration was performed selectively rather than evaluating the entire search space.

The strongest configuration was discovered at approximately:

**Oracle Call 52**

A repeated evaluation at:

**Oracle Call 53**

returned the same best loss:

**7.6139588414**

Therefore, instead of evaluating:

**2304 configurations**

the search obtained a strong result using approximately:

**53 Oracle evaluations**

This corresponds to exploring only about:

**2.3% of the complete search space**

while still substantially improving the initial model.

---

## Why Grid Search Was Avoided

The complete search space contained:

**2304 configurations**

An exhaustive Grid Search would have evaluated every configuration regardless of whether previous experiments already indicated that some regions were poor.

For example, the experiment:

**eta0 = 0.0001 → Loss = 48.5448701129**

provided strong evidence that this setting was poor in the tested configuration.

Similarly:

**average = True → Loss = 12.4151212076**

was clearly worse than the current best configuration.

Instead of continuing to explore such settings blindly, the search concentrated API calls on more promising areas.

---

## Greedy Coordinate Search

The first major strategy used was similar to **greedy coordinate descent**.

One hyperparameter was modified while the remaining parameters were kept fixed.

If the modification improved the loss, the change was accepted.

Otherwise, it was rejected.

This method was useful because it was:

- Simple to implement
- Easy to interpret
- API efficient
- Suitable when optimization libraries were unavailable

However, its main limitation is that hyperparameters interact.

For this reason, the later stages also explored selected combinations of parameters rather than relying entirely on independent one-at-a-time tuning.

---

## Coarse-to-Fine Exploration

The search was performed in a coarse-to-fine manner.

Instead of immediately testing every possible value of a parameter, representative values were evaluated first.

Poor regions were discarded while promising regions received additional exploration.

For example, the `alpha` parameter contained eight possible values.

Testing all eight values under every possible combination would have consumed many API calls.

Instead, the search concentrated on values that showed improvement and then explored nearby or interacting configurations.

This reduced unnecessary evaluations.

---

## Importance of Hyperparameter Interaction

One of the most important observations from the experiment was that hyperparameters should not always be considered independently.

A good example was `eta0`.

Early in the search:

**eta0 = 0.001**

performed better than the original baseline.

However, after other parameters had been improved:

**eta0 = 0.05**

produced an even lower loss of:

**7.7045678683**

This shows that the best value of one hyperparameter depends on the values of other hyperparameters.

Therefore, the search evolved from simple one-parameter experimentation to selected joint and pairwise exploration.

---

## Best Result

The lowest loss obtained during the optimization was:

# **7.6139588414**

This result was reached after approximately:

# **52 Oracle calls**

and was confirmed again on the next evaluation.

One important component of the final improvement was:

**learning_rate = optimal**

with a promising configuration also using:

**eta0 = 0.05**

---

## Final Outcome

The optimization successfully reduced the loss from:

**7.7731686227**

to:

**7.6139588414**

for a total reduction of:

**0.1592097813**

The search achieved this without evaluating the complete set of 2304 configurations.

Only a small fraction of the search space was explored, with each Oracle result being used to guide the next experiment.

---

## Key Learnings

The main lesson from this task was that hyperparameter optimization does not always require exhaustive search.

When model evaluations are expensive, it is more useful to think of every evaluation as an experiment.

Each API call should either:

- Improve the current best solution,
- Eliminate an unpromising region,
- Provide information about a hyperparameter,
- Or test an important interaction between parameters.

The experiment also showed that simple greedy tuning is useful for quickly finding improvements, but interactions between hyperparameters must eventually be considered.

The final strategy therefore combined:

**Greedy Search + Coarse-to-Fine Exploration + Pairwise Parameter Testing**

This allowed the SGDRegressor configuration to be improved substantially while keeping the number of Oracle API calls far below a complete Grid Search.