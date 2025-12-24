# t-SNE (t-Distributed Stochastic Neighbor Embedding) - Deep Dive Explanation

## Table of Contents

1. [Introduction & Intuition](#introduction--intuition)
2. [High-Dimensional Similarities (P Matrix)](#high-dimensional-similarities-p-matrix)
3. [Perplexity & Binary Search](#perplexity--binary-search)
4. [Low-Dimensional Similarities (Q Matrix)](#low-dimensional-similarities-q-matrix)
5. [KL Divergence & Gradient Descent](#kl-divergence--gradient-descent)
6. [Complete Optimization Loop](#complete-optimization-loop)
7. [Step-by-Step Flow Examples](#step-by-step-flow-examples)
8. [Internal Changes During Training](#internal-changes-during-training)

---

## Introduction & Intuition

### What is t-SNE?

t-SNE (t-Distributed Stochastic Neighbor Embedding) is a **dimensionality reduction technique** that converts high-dimensional data into a low-dimensional representation (typically 2D or 3D) for visualization. Unlike PCA which preserves global variance, t-SNE focuses on preserving **local neighborhood structure**.

### The Core Problem

Imagine you have data in 64 dimensions (like 8×8 digit images). How do you visualize it?

**PCA approach**: Find directions of maximum variance (global structure)

**t-SNE approach**: Keep similar points close together (local structure)

### Key Terms

- **Perplexity**: Related to the number of nearest neighbors (typically 5-50). Perplexity = $2^{H(P)}$ where $H$ is Shannon entropy.
- **P Matrix**: Pairwise similarities in **high-dimensional space** using Gaussian kernel
- **Q Matrix**: Pairwise similarities in **low-dimensional space** using t-distribution
- **KL Divergence**: Measures difference between P and Q distributions
- **Early Exaggeration**: Multiplies P by a factor (e.g., 4) in early iterations to form tight clusters

### The t-SNE Workflow

```
High-Dim Data (64D)
        ↓
    Compute P Matrix
   (Gaussian kernel)
        ↓
    Initialize Y (2D)
        ↓
    Optimization Loop
    (Gradient Descent)
        ↓
    Compute Q Matrix
   (t-distribution)
        ↓
    Minimize KL(P||Q)
        ↓
  Low-Dim Embedding (2D)
```

---

## High-Dimensional Similarities (P Matrix)

### Mathematical Foundation

For each point **i**, we compute the conditional probability $p_{j|i}$ that point **i** would pick point **j** as its neighbor:

$$p_{j|i} = \frac{\exp\left(-\frac{||x_i - x_j||^2}{2\sigma_i^2}\right)}{\sum_{k \neq i} \exp\left(-\frac{||x_i - x_k||^2}{2\sigma_i^2}\right)}$$

**Where:**

- $x_i$ = high-dimensional point i (e.g., 64-dimensional vector for 8×8 images)
- $||x_i - x_j||^2$ = squared Euclidean distance between points i and j
- $\sigma_i$ = **bandwidth parameter for point i** (unique per point!)
- The numerator is the **Gaussian kernel**

### Efficient Distance Computation

Computing pairwise distances with nested loops is $O(n^2 \cdot d)$. We use a mathematical identity:

$$||a - b||^2 = ||a||^2 + ||b||^2 - 2a \cdot b$$

This allows us to compute the entire distance matrix in $O(n^2)$:

```python
def _compute_pairwise_distances(self, X):
    # ||xi||² for each point
    sum_X = np.sum(np.square(X), axis=1)

    # Broadcast: ||xi||² + ||xj||² - 2*xi·xj
    distances = -2 * np.dot(X, X.T) + sum_X[:, None] + sum_X[None, :]

    # Diagonal should be 0 (distance from point to itself)
    np.fill_diagonal(distances, 0)

    # Avoid negative values from floating point errors
    return np.maximum(distances, 0)
```

### Symmetrization

After computing conditional probabilities $p_{j|i}$ for all points, we make the matrix symmetric:

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

This ensures:
- $p_{ij} = p_{ji}$ (symmetry)
- $\sum_{i,j} p_{ij} = 1$ (normalization)

### Implementation Breakdown

```python
class TSNE:
    def __init__(self, n_components=2, perplexity=30, n_iter=1000, learning_rate=200):
        self.n_components = n_components      # Output dimensions (usually 2)
        self.perplexity = perplexity          # Effective neighbors (5-50)
        self.n_iter = n_iter                  # Optimization iterations
        self.learning_rate = learning_rate    # Gradient step size
```

#### The `fit_transform` method

```python
def fit_transform(self, X):
    # Step 1: Compute high-dimensional similarities
    P = self._compute_pairwise_similarities_high_dim(X)

    # Step 2: Initialize low-dimensional embedding with small random values
    n_samples = X.shape[0]
    Y = np.random.randn(n_samples, self.n_components) * 1e-4

    # Step 3: Optimize Y using gradient descent
    Y = self._optimize(Y, P)

    return Y
```

#### Computing P matrix

```python
def _compute_pairwise_similarities_high_dim(self, X):
    n_samples = X.shape[0]
    P = np.zeros((n_samples, n_samples))

    # Compute pairwise distances
    distances = self._compute_pairwise_distances(X)

    # For each point, find optimal sigma via binary search
    for i in range(n_samples):
        sigma = self._find_optimal_sigma(distances[i], self.perplexity)

        # Compute p_j|i using Gaussian kernel
        P[i] = np.exp(-distances[i] / (2 * sigma ** 2))
        P[i] = P[i] / (np.sum(P[i]) + 1e-10)  # Normalize

    # Symmetrize
    P = (P + P.T) / (2 * n_samples)

    return P
```

---

## Perplexity & Binary Search

### Understanding Perplexity

Perplexity is defined as:

$$\text{Perp}(P_i) = 2^{H(P_i)}$$

Where $H(P_i)$ is the **Shannon entropy**:

$$H(P_i) = -\sum_j p_{j|i} \log_2(p_{j|i})$$

**Interpretation:**

- Perplexity = effective number of neighbors
- Low perplexity (5): Focus on very local structure
- Medium perplexity (30): Balance local and global (recommended)
- High perplexity (50+): Focus more on global structure

### Why Binary Search?

The relationship between $\sigma$ and perplexity:

| $\sigma$ | Distribution | Perplexity |
|----------|-------------|------------|
| Small | Peaked (few neighbors matter) | Low |
| Large | Flat (many neighbors matter) | High |

Since perplexity increases monotonically with $\sigma$, we can use **binary search** to find the $\sigma$ that matches our target perplexity.

### Binary Search Implementation

```python
def _find_optimal_sigma(self, distances, target_perplexity, tol=1e-5, max_iter=50):
    # Binary search bounds
    sigma_min = 1e-10   # Very small → very low perplexity
    sigma_max = 1e10    # Very large → very high perplexity

    for _ in range(max_iter):
        sigma = (sigma_min + sigma_max) / 2

        # Compute probabilities with current sigma
        p = np.exp(-distances / (2 * sigma ** 2))
        p = p / (np.sum(p) + 1e-10)  # Normalize

        # Check perplexity
        perplexity = self._compute_perplexity(p)

        if perplexity > target_perplexity:
            sigma_max = sigma      # Too high, decrease sigma
        else:
            sigma_min = sigma      # Too low, increase sigma

        if np.abs(perplexity - target_perplexity) < tol:
            break

    return sigma
```

### Computing Perplexity

```python
def _compute_perplexity(self, p):
    # Shannon entropy: H = -sum(p * log2(p))
    entropy = -np.sum(p * np.log2(p + 1e-10))

    # Perplexity = 2^entropy
    return 2 ** entropy
```

### What Happens Internally: Perplexity Search

**Scenario:** We have 5 points with distances from point 0: `[0, 1, 2, 4, 8]`

**Target perplexity:** 5

---

**Iteration 1: Try σ = 1**

Compute raw similarities:
```
exp(-[0, 1, 2, 4, 8] / 2) = exp([0, -0.5, -1, -2, -4])
                           = [1.0, 0.607, 0.368, 0.135, 0.018]
```

Normalize (divide by sum):
```
p = [1.0, 0.607, 0.368, 0.135, 0.018] / 2.128
  = [0.470, 0.285, 0.173, 0.063, 0.008]
```

Compute entropy:
```
H = -sum(p * log2(p))
  = -(0.470 * -1.09 + 0.285 * -1.81 + 0.173 * -2.53 + 0.063 * -3.98 + 0.008 * -6.96)
  = -(-0.51 - 0.52 - 0.44 - 0.25 - 0.06)
  = 1.78 bits
```

Perplexity:
```
Perp = 2^1.78 = 3.42
```

**3.42 < 5** → Need larger σ

---

**Iteration 2: Try σ = 3**

Compute raw similarities:
```
exp(-[0, 1, 2, 4, 8] / 18) = exp([0, -0.056, -0.111, -0.222, -0.444])
                            = [1.0, 0.946, 0.895, 0.801, 0.641]
```

Normalize:
```
p = [1.0, 0.946, 0.895, 0.801, 0.641] / 4.283
  = [0.233, 0.221, 0.209, 0.187, 0.150]
```

Compute entropy:
```
H = -sum(p * log2(p))
  = -(0.233 * -2.10 + 0.221 * -2.18 + 0.209 * -2.26 + 0.187 * -2.42 + 0.150 * -2.74)
  = 2.30 bits
```

Perplexity:
```
Perp = 2^2.30 = 4.92
```

**4.92 ≈ 5** → Close enough!

---

**Final:** σ₀ = 3 is optimal for point 0

Each point gets its own σ based on its local neighborhood density!

---

## Low-Dimensional Similarities (Q Matrix)

### The t-Distribution Kernel

In low-dimensional space, we use the **t-distribution with 1 degree of freedom** (Cauchy distribution):

$$q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l} (1 + ||y_k - y_l||^2)^{-1}}$$

**Key differences from high-dimensional space:**

| Aspect | High-Dim (P) | Low-Dim (Q) |
|--------|-------------|-------------|
| Distribution | Gaussian | t-distribution (ν=1) |
| Bandwidth | Per-point σᵢ | Single (implicit) |
| Tail behavior | exp(-x²) drops fast | (1+x²)⁻¹ drops slow |

### Why t-Distribution?

**The crowding problem:** In high dimensions, there's much more volume. When projecting to 2D, points get "crowded" together.

The t-distribution has **heavier tails**, meaning:
- Points further apart still have some "repulsion"
- Prevents all points from collapsing into a single region
- Creates more natural, spread-out visualizations

### Implementation

```python
def _compute_low_dim_similarities(self, Y):
    # Compute pairwise distances in low-dimensional space
    distances = self._compute_pairwise_distances(Y)

    # t-distribution kernel: (1 + distance²)^(-1)
    numerator = (1 + distances) ** (-1)

    # Diagonal = 0 (no self-similarity)
    np.fill_diagonal(numerator, 0)

    # Normalize
    Q = numerator / np.sum(numerator)

    return Q
```

### Comparing Gaussian vs t-Distribution

| Distance | Gaussian: exp(-d²) | t-dist: (1+d²)⁻¹ |
|----------|-------------------|------------------|
| 0 | 1.00 | 1.00 |
| 1 | 0.37 | 0.50 |
| 2 | 0.02 | 0.20 |
| 3 | 0.00 | 0.10 |

The t-distribution maintains higher similarity at larger distances!

---

## KL Divergence & Gradient Descent

### KL Divergence (Cost Function)

We measure the difference between P (high-dim) and Q (low-dim) using **Kullback-Leibler divergence**:

$$KL(P||Q) = \sum_{i \neq j} p_{ij} \log\left(\frac{p_{ij}}{q_{ij}}\right)$$

**Properties:**

1. **KL ≥ 0**: Minimum is 0 when P = Q
2. **Not symmetric**: KL(P||Q) ≠ KL(Q||P)
3. **Heavy penalty**: When p_ij is high but q_ij is low, the penalty is huge

**Intuition:** We care most about preserving **high-probability neighbors**. If points are close in high-dim (high p_ij), they MUST be close in low-dim (high q_ij).

### The Gradient

To minimize KL divergence, we compute the gradient:

$$\frac{\partial C}{\partial y_i} = 4 \sum_j (p_{ij} - q_{ij}) (y_i - y_j) (1 + ||y_i - y_j||^2)^{-1}$$

**Interpretation:**

- $(p_{ij} - q_{ij})$ = **force direction**
  - Positive: Points too far in low-dim → **attract**
  - Negative: Points too close in low-dim → **repel**
- $(y_i - y_j)$ = **direction vector**
- $(1 + ||y_i - y_j||^2)^{-1}$ = **weight based on distance**

### Implementation

```python
def _compute_gradient(self, P, Q, Y):
    n_samples = Y.shape[0]
    grad = np.zeros_like(Y)

    # Pairwise differences in Y
    Y_diff = Y[:, None, :] - Y[None, :, :]  # Shape: (n, n, n_components)

    # Distances in low-dim
    distances = self._compute_pairwise_distances(Y)

    # Weights: (1 + distance²)^(-1)
    weights = (1 + distances) ** (-1)

    # Force: P - Q
    PQ_diff = P - Q

    # Compute gradient for each point
    for i in range(n_samples):
        # Sum over all j
        grad[i] = 4 * np.sum(
            PQ_diff[i, :, None] * weights[i, :, None] * Y_diff[i, :, :],
            axis=0
        )

    return grad
```

### KL Divergence Computation

```python
def _compute_kl_divergence(self, P, Q):
    # KL = sum(p * log(p/q))
    # Add epsilon to avoid log(0)
    kl = np.sum(P * np.log((P + 1e-10) / (Q + 1e-10)))
    return kl
```

---

## Complete Optimization Loop

### The Training Process

```python
def _optimize(self, Y, P):
    n_samples = P.shape[0]

    # Early exaggeration: amplify P in early iterations
    early_exaggeration = 4
    exaggeration_cutoff = 100

    for iteration in range(self.n_iter):
        # Step 1: Compute Q matrix (low-dim similarities)
        Q = self._compute_low_dim_similarities(Y)

        # Step 2: Apply early exaggeration
        if iteration < exaggeration_cutoff:
            P_eff = P * early_exaggeration
        else:
            P_eff = P

        # Step 3: Compute gradient
        grad = self._compute_gradient(P_eff, Q, Y)

        # Step 4: Update with momentum
        if iteration == 0:
            momentum = np.zeros_like(Y)

        # Momentum increases during training
        if iteration < 250:
            momentum_coeff = 0.5
        else:
            momentum_coeff = 0.8

        # Update momentum and Y
        momentum = momentum_coeff * momentum - self.learning_rate * grad
        Y = Y + momentum

        # Step 5: Center Y at origin
        Y = Y - np.mean(Y, axis=0)

        # Print progress
        if iteration % 100 == 0:
            cost = self._compute_kl_divergence(P_eff, Q)
            print(f"Iteration {iteration}: KL divergence = {cost:.4f}")

    return Y
```

### Why Momentum?

Momentum helps:
1. **Escape local minima**: Accumulates velocity to push through small barriers
2. **Faster convergence**: Builds up speed in consistent directions
3. **Smoother optimization**: Dampens oscillations

### Why Early Exaggeration?

In early iterations, we multiply P by a factor (usually 4):

$$P_{eff} = 4 \times P$$

This:
- Creates **stronger attractive forces** between similar points
- Points are pulled together more aggressively
- Forms **tight, well-separated clusters**
- Is turned off after ~100 iterations

---

## Step-by-Step Flow Examples

### Complete Worked Example (3 points)

Let's trace through a complete t-SNE iteration with 3 points in 2D → 1D.

**High-dimensional data (2D):**
```
x₀ = [0, 0]
x₁ = [1, 0]
x₂ = [3, 0]
```

---

#### **Step 1: Compute Distance Matrix**

```
||x₀ - x₀||² = 0² + 0² = 0
||x₀ - x₁||² = 1² + 0² = 1
||x₀ - x₂||² = 3² + 0² = 9
||x₁ - x₂||² = 2² + 0² = 4
```

Distance matrix D:
```
     x₀   x₁   x₂
x₀ [  0     1     9  ]
x₁ [  1     0     4  ]
x₂ [  9     4     0  ]
```

---

#### **Step 2: Compute P Matrix (σ = 1 for simplicity)**

For point 0:
```
p_1|0 = exp(-1/2) / (exp(0) + exp(-1/2) + exp(-9/2))
      = 0.607 / (1 + 0.607 + 0.011)
      = 0.607 / 1.618
      = 0.375

p_2|0 = exp(-9/2) / 1.618
      = 0.011 / 1.618
      = 0.007
```

Similarly for other points, then symmetrize:

```
P ≈ [  0      0.15    0.02
       0.15    0      0.08
       0.02    0.08    0    ]
```

(Note: Values are approximate for illustration)

---

#### **Step 3: Initialize Low-Dimensional Embedding**

```
y₀ =  0.001
y₁ = -0.002
y₂ =  0.0003
```

(Small random values from N(0, 10⁻⁴))

---

#### **Step 4: Compute Q Matrix**

Low-dimensional distances:
```
||y₀ - y₁||² = (0.001 - (-0.002))² = 0.003² = 0.000009
||y₀ - y₂||² = (0.001 - 0.0003)² = 0.0007² = 0.00000049
||y₁ - y₂||² = (-0.002 - 0.0003)² = 0.0023² = 0.00000529
```

t-distribution kernel:
```
q_01 = (1 + 0.000009)^(-1) / sum ≈ 0.25
q_02 = (1 + 0.00000049)^(-1) / sum ≈ 0.25
q_12 = (1 + 0.00000529)^(-1) / sum ≈ 0.25
```

(Note: In early iterations with all points close, Q is nearly uniform)

---

#### **Step 5: Compute Gradient for y₀**

$$\frac{\partial C}{\partial y_0} = 4 \sum_j (p_{0j} - q_{0j}) \frac{y_0 - y_j}{1 + ||y_0 - y_j||^2}$$

For j = 1:
```
term_1 = (p_01 - q_01) * (y_0 - y_1) / (1 + ||y_0 - y_1||²)
      = (0.15 - 0.25) * (0.001 - (-0.002)) / (1 + 0.000009)
      = (-0.10) * (0.003) / 1.000009
      = -0.00030
```

For j = 2:
```
term_2 = (p_02 - q_02) * (y_0 - y_2) / (1 + ||y_0 - y_2||²)
      = (0.02 - 0.25) * (0.001 - 0.0003) / (1 + 0.00000049)
      = (-0.23) * (0.0007) / 1.00000049
      = -0.00016
```

Total gradient:
```
grad_0 = 4 * (term_1 + term_2)
      = 4 * (-0.00030 - 0.00016)
      = 4 * (-0.00046)
      = -0.00184
```

---

#### **Step 6: Update y₀ with Momentum**

```
momentum_new = 0.5 * 0 - 200 * (-0.00184)
             = 0 + 0.368
             = 0.368

y_0_new = 0.001 + 0.368
        = 0.369
```

Repeat for all points, for all iterations!

---

### What This Means

**Negative gradient** (-0.00184) means:
- Current q_01 > p_01 (0.25 > 0.15)
- Points are too close in low-dim
- Need to **move y₀ away from y₁**

But wait - the update is **positive** (0.368). Why?

Because:
- y₁ = -0.002 (negative position)
- y₀ = 0.001 (positive position)
- To move y₀ **away** from y₁, we make y₀ **more positive**

---

## Internal Changes During Training

### Visualization of the Training Process

Imagine we're training on the digits dataset. Here's what happens internally:

---

### **Before Training (Iteration 0)**

```
Y: Random small values from N(0, 10⁻⁴)
   All points clustered near origin

Q: Nearly uniform (all points have similar distances)

KL divergence: High (Q doesn't match P at all)

Forces: Chaotic, large magnitude
```

The embedding looks like a **random blob** at the origin.

---

### **Early Training (Iterations 0-100)**

**Early exaggeration active: P_eff = 4 × P**

```
Y: Points rapidly moving apart
   Clusters starting to form

Q: Becoming more structured
   Some similarities higher, some lower

KL divergence: Decreasing rapidly
   From ~2.5 to ~1.0

Forces: Strong attraction between similar points
       Points of same digit pulling together
```

**What's happening:**

- P is amplified (×4), so attractive forces are strong
- Points with high p_ij (similar digits) attract each other
- The embedding **explodes outward** from origin
- **Tight clusters** form quickly

**Visualization:** Clusters are very tight and well-separated.

---

### **Mid Training (Iterations 100-500)**

**Early exaggeration OFF: P_eff = P**

```
Y: Clusters stabilizing
   Refining internal structure

Q: Matching P better
   Local neighborhoods preserved

KL divergence: Decreasing slowly
   From ~1.0 to ~0.7

Forces: Balanced attraction/repulsion
       Fine-tuning positions within clusters
```

**What's happening:**

- Attraction forces are now normal (no exaggeration)
- Clusters may **expand slightly**
- Within-cluster structure refines
- Points settle into their local neighborhoods

**Visualization:** Clusters are more natural-looking, with some internal structure visible.

---

### **Late Training (Iterations 500-1000)**

```
Y: Stable
   Only small adjustments

Q: Very close to P
   Local structure well-preserved

KL divergence: Converged
   ~0.6 to 0.7 (minimal change)

Forces: Small, balanced
       Mostly maintaining structure
```

**What's happening:**

- Momentum is high (0.8), but gradients are small
- Points make tiny adjustments
- The embedding has **converged**
- Running more iterations gives diminishing returns

**Visualization:** Final t-SNE plot with well-separated digit clusters.

---

### **After Training (Converged)**

```
Y: Optimal low-dimensional embedding

Structure preserved:
  - Similar digits are close
  - Different digits are far apart
  - Local neighborhoods maintained

Structure NOT preserved:
  - Distances between clusters
  - Relative cluster sizes
  - Global geometry
```

---

### Loss (KL Divergence) Over Time

```
Iteration | KL Divergence
----------|---------------
     0    |    2.4567
    100   |    1.0234  (early exaggeration ends)
    200   |    0.8765
    300   |    0.7890
    400   |    0.7234
    500   |    0.6876
    600   |    0.6543
    700   |    0.6321
    800   |    0.6178
    900   |    0.6089
   1000   |    0.6032
```

Notice the **steep drop** in early iterations (early exaggeration), followed by **gradual convergence**.

---

### Perplexity Effects on Internal Structure

**Low Perplexity (5-15):**
```
P: Each point focuses on very few neighbors
   High p_ij only for closest points

Training: More local clusters form
         Some global structure lost

Result: Many small, tight clusters
       May appear fragmented
```

**Medium Perplexity (30-50):**
```
P: Balance of local and some global
   Each point considers ~30-50 neighbors

Training: Balanced attraction forces
         Both local and global structure

Result: Well-formed clusters
       Global structure visible
       (Recommended)
```

**High Perplexity (50+):**
```
P: Each point considers many neighbors
   Smoother probability distribution

Training: More global structure preserved
         Some local details lost

Result: Larger, merged clusters
       May lose fine-grained structure
```

---

## Parameter Effects

### Perplexity

| Value | Effect | Use Case |
|-------|--------|----------|
| 5-15 | Very local | Many small clusters, fragmented |
| 30-50 | Balanced | Default choice, works well |
| 50+ | Global | Larger clusters, smoother |

### Learning Rate

| Value | Effect | Use Case |
|-------|--------|----------|
| < 100 | Too slow | Wastes computation |
| 100-500 | Good | Typical range |
| > 1000 | Unstable | May cause issues |

### Number of Iterations

| Value | Effect |
|-------|--------|
| < 500 | May not converge |
| 1000 | Standard choice |
| > 1000 | Diminishing returns |

---

## Common Pitfalls

### 1. Interpreting Cluster Sizes

**Wrong:** "This tight cluster means these points are more similar."

**Correct:** Cluster sizes in t-SNE are **not meaningful**. Only local neighborhoods matter.

### 2. Interpreting Cluster Distances

**Wrong:** "These clusters are far apart, so they're fundamentally different."

**Correct:** Distances between clusters are **not meaningful** in t-SNE.

### 3. Using t-SNE for Feature Engineering

**Wrong:** "I'll use t-SNE embeddings as features for my classifier."

**Correct:** t-SNE is **stochastic and non-linear**. Use PCA or other methods for feature engineering.

### 4. Wrong Perplexity

**Too low (5):** Data looks fragmented, many tiny clusters

**Too high (100):** Data looks merged, loss of detail

**Solution:** Always try multiple perplexity values!

### 5. Not Standardizing Data

t-SNE is **distance-based**. If features have different scales:
- Large-scale features dominate distances
- Results are biased

**Solution:** Always `StandardScaler()` before t-SNE!

---

## Comparison with Other Methods

| Method | What It Preserves | Global Structure | Speed | Deterministic |
|--------|------------------|------------------|-------|---------------|
| **PCA** | Large variance directions | Yes | Fast | Yes |
| **t-SNE** | Local neighborhoods | Weak | Slow (O(n²)) | No |
| **UMAP** | Local + some global | Moderate | Fast | No |

### When to Use Each

- **PCA**: Quick overview, feature engineering, global structure
- **t-SNE**: Beautiful visualizations, exploring local structure
- **UMAP**: Fast alternative to t-SNE with better global structure

---

## Key Takeaways

### The Training Process in One Sentence

**Repeatedly move points in low-dimensional space so that their pairwise similarities (using t-distribution) match their pairwise similarities in high-dimensional space (using Gaussian kernel).**

### Why t-SNE Works

1. **Local structure focus**: Preserves neighborhoods, not global geometry
2. **t-distribution**: Heavy tails prevent crowding in low-dim
3. **Perplexity**: Adapts to local density (each point gets its own σ)
4. **Early exaggeration**: Forms tight clusters quickly
5. **Momentum**: Helps escape local minima and speeds convergence

### What t-SNE Tells You

✅ **Meaningful:**
- Points close in t-SNE → likely neighbors in high-dim
- Well-separated clusters → genuine structure
- Local neighborhoods preserved

❌ **NOT Meaningful:**
- Cluster sizes
- Distances between clusters
- Global geometry
- Absolute positions

### When to Use t-SNE

- Exploring high-dimensional data visually
- Understanding cluster structure
- Debugging classification models
- Finding patterns in unlabeled data

---

## Conclusion

You've implemented **t-SNE from scratch** with:

- High-dimensional similarity computation (Gaussian kernel, per-point σ)
- Perplexity-based bandwidth selection (binary search)
- Low-dimensional similarity computation (t-distribution)
- KL divergence minimization (gradient descent with momentum)
- Early exaggeration for tight cluster formation

The key insight is: **t-SNE converts distances to probabilities in both high and low dimensions, then optimizes the low-dimensional layout to match the high-dimensional probabilities.**

Understanding the flow means understanding this cycle:

1. **Convert high-dim distances to probabilities** (P matrix, Gaussian)
2. **Convert low-dim distances to probabilities** (Q matrix, t-distribution)
3. **Measure difference** (KL divergence)
4. **Move points to minimize difference** (gradient descent)
5. **Repeat until convergence**

The internal changes are the gradual evolution of Y from random noise to a meaningful 2D representation where similar points are close together.

Now you truly understand not just **what** your t-SNE code does, but **why** and **how** it works! 🎉

---

## References

1. van der Maaten, L., & Hinton, G. (2008). "Visualizing Data using t-SNE" [Original paper](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)

2. Distill: "How to Use t-SNE Effectively" [Interactive guide](https://distill.pub/2016/misread-tsne/)

3. scikit-learn documentation on [t-SNE](https://scikit-learn.org/stable/modules/manifold.html#t-sne)

---

*Generated for educational purposes - t-SNE implementation from scratch with detailed mathematical explanations and step-by-step worked examples.*