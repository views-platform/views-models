# --- Important Note on Darts NBEATSModel (generic_architecture=True) ---
#
# Observation: It has been empirically confirmed that for the Darts NBEATSModel
# using `generic_architecture=True`, configurations where `num_stacks` and
# `num_blocks` are swapped (e.g., (num_stacks=1, num_blocks=2) vs. (num_stacks=2, num_blocks=1))
# will produce **identical computational results**.
#
# Reason: The NBEATS architecture processes blocks and stacks in a sequential
# manner, passing residuals and summing forecasts. The total effective computational
# depth, or the total number of blocks processed in series, is `num_stacks * num_blocks`.
# Therefore, any two configurations where the product of these values is the same
# (e.g., 1*2=2 and 2*1=2) result in an identical computational graph and will yield
# the exact same outputs given identical inputs and initialization.
#
# Recommendation: Be mindful of this property when designing hyperparameter search
# spaces to avoid redundant experiments or misinterpreting the effects of these
# hyperparameters when their product remains constant.
