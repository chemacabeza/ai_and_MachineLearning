import numpy as np

def bayesian_coin_inference():
    """
    Bayesian inference for a biased coin.
    We discretize the bias parameter theta into a grid and update
    the posterior after each observed flip.
    """
    # Discretize theta (coin bias) into 100 values from 0 to 1
    theta_grid = np.linspace(0, 1, 100)

    # Start with a uniform prior: "I have no idea what the bias is"
    prior = np.ones_like(theta_grid) / len(theta_grid)

    # True coin bias (unknown to the learner)
    true_bias = 0.7
    np.random.seed(42)

    # Simulate 50 coin flips
    flips = np.random.binomial(1, true_bias, size=50)

    print("=" * 60)
    print("BAYESIAN COIN INFERENCE")
    print("=" * 60)
    print(f"True coin bias: {true_bias}")
    print(f"Prior: Uniform (no initial assumption)")
    print("-" * 60)

    posterior = prior.copy()

    milestones = [1, 5, 10, 20, 50]
    heads_count = 0

    for i, flip in enumerate(flips, 1):
        heads_count += flip

        # Likelihood: P(flip | theta)
        if flip == 1:  # heads
            likelihood = theta_grid
        else:  # tails
            likelihood = 1 - theta_grid

        # Bayesian update: posterior ∝ likelihood × prior
        posterior = likelihood * posterior
        posterior = posterior / posterior.sum()  # normalize

        if i in milestones:
            map_estimate = theta_grid[np.argmax(posterior)]
            mle_estimate = heads_count / i if i > 0 else 0.5
            confidence = posterior.max()

            print(f"\nAfter {i:2d} flips ({heads_count}H, {i - heads_count}T):")
            print(f"  MLE estimate:  {mle_estimate:.4f}")
            print(f"  MAP estimate:  {map_estimate:.4f}")
            print(f"  Peak confidence: {confidence:.4f}")

            # Show posterior distribution as ASCII bar chart
            # Downsample to 20 bins for display
            bins = 20
            bin_size = len(theta_grid) // bins
            print(f"  Posterior distribution:")
            for b in range(bins):
                start = b * bin_size
                end = start + bin_size
                bar_val = posterior[start:end].sum()
                bar = "█" * int(bar_val * 200)
                label = f"  θ={theta_grid[start]:.2f}-{theta_grid[end-1]:.2f}"
                if bar:
                    print(f"    {label} |{bar}")

    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print("With few flips, the prior matters a lot (MAP ≠ MLE).")
    print("With many flips, data dominates and MAP ≈ MLE.")
    print("Both converge toward the true bias as data increases.")
    print("=" * 60)

if __name__ == "__main__":
    bayesian_coin_inference()
