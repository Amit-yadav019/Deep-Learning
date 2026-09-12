# Karpathy Ep1 — Micrograd Complete Notes
> Building Neural Networks + Backpropagation from Scratch

---

## What Is Micrograd?

Micrograd is a tiny **autograd engine** — a system that automatically computes gradients.

```
Auto  = automatic
Grad  = gradient (derivative)
Autograd = automatically compute derivatives
```

**Why build it?**
```
PyTorch does this for you automatically.
But if you don't understand HOW it works → you can't debug training
You won't know why loss explodes, why gradients vanish, why model doesn't learn.

Micrograd = understanding what happens INSIDE PyTorch's .backward()
```

---

## Core Idea — What Problem Are We Solving?

```
You have a neural network with millions of weights (w).
You want to find the values of w that minimise loss.
To minimise loss → use gradient descent:
    w = w - learning_rate × (dLoss/dw)

The question: HOW do you compute dLoss/dw for every single weight?
Answer: Backpropagation through the computation graph.
Micrograd builds this computation graph and propagates gradients backward.
```

---

## 1. The Value Class — Every Line Explained

```python
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data      = data          # The actual number stored (e.g. 2.0, -3.5)
        self.grad      = 0.0           # dLoss/d(this value). Starts 0, filled by backward()
        self._backward = lambda: None  # Function that computes gradient. Empty by default.
        self._prev     = set(_children) # Set of Values that CREATED this Value
        self._op       = _op           # Which operation created this (+, *, tanh etc.)
        self.label     = label         # Optional name for visualisation
```

**Why `_children`?**
```python
a = Value(2.0)
b = Value(3.0)
c = a + b      # c._prev = {a, b} ← c was created FROM a and b
               # This is how we trace the computation graph backward
```

**Why `grad = 0.0` initially?**
```
Gradient = how much does Loss change if I increase THIS value slightly?
Before running backward() → we don't know → start at 0
After backward() → filled with actual gradient values
```

**Why `_backward = lambda: None`?**
```
For leaf nodes (input data, weights) → no children → nothing to propagate back
For operation nodes (+, *, tanh) → _backward is replaced with the chain rule formula
```

---

## 2. Forward Pass — Computing Output

```python
a = Value(2.0, label='a')
b = Value(3.0, label='b')
c = a * b                   # c.data = 6.0
d = c + Value(1.0)          # d.data = 7.0
e = d.tanh()                # e.data = tanh(7.0) ≈ 0.9999
```

**What the computation graph looks like:**
```
a(2.0) ──┐
         × ──→ c(6.0) ──┐
b(3.0) ──┘              + ──→ d(7.0) ──→ tanh ──→ e(0.9999)
                  1.0 ──┘
```

This is the **forward pass** — data flows left to right, computing the output.

---

## 3. Derivatives — What They Mean

```
Derivative of f at x = how much does f(x) change when x increases by a tiny amount h?

dLoss/dw = if I increase w by 0.001, how much does Loss change?

Positive gradient → increasing w increases Loss → decrease w
Negative gradient → increasing w decreases Loss → increase w
Zero gradient    → changing w doesn't affect Loss → w is already optimal
```

**Numerical approximation (for verification):**
```python
def numerical_grad(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# If analytical (backprop) gradient ≈ numerical gradient → backprop is correct
```

---

## 4. The Chain Rule — Core of Backpropagation

```
If e = tanh(d) and d = c + 1 and c = a × b:

de/da = de/dd × dd/dc × dc/da
      ↑          ↑          ↑
  tanh deriv  add deriv   mul deriv
```

**In English:** The gradient flows backward through each operation, multiplied at each step.

```
Start at output: de/de = 1.0 (gradient of output w.r.t. itself)
Backward through tanh: dd/dd = de/de × (1 - tanh²(d))
Backward through +: dc/dc = dd/dd × 1
Backward through ×: da/da = dc/dc × b.data
                    db/db = dc/dc × a.data
```

---

## 5. Backward Functions — Each Operation Explained

### Addition: `c = a + b`
```python
def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        # d(a+b)/da = 1 → gradient passes through unchanged
        # d(a+b)/db = 1 → gradient passes through unchanged
        self.grad  += out.grad × 1  →  self.grad  += out.grad
        other.grad += out.grad × 1  →  other.grad += out.grad

    out._backward = _backward
    return out
```

**Why `+=` not `=`?**
```
A variable can be used in multiple operations.
Example: a = Value(2.0);  b = a + a  (a used twice)
Gradient accumulates from BOTH uses.
+= prevents overwriting the gradient from the first use.
```

**Intuition:** Addition is like a "gradient router" — same gradient goes to both inputs.

---

### Multiplication: `c = a * b`
```python
def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        # d(a×b)/da = b → gradient scaled by OTHER value
        # d(a×b)/db = a → gradient scaled by SELF value
        self.grad  += other.data × out.grad
        other.grad += self.data  × out.grad

    out._backward = _backward
    return out
```

**Intuition:** If a=2, b=100, loss=a×b:
- Changing a by 1 → loss changes by 100 (gradient of a = 100 = b)
- Changing b by 1 → loss changes by 2 (gradient of b = 2 = a)

---

### Power: `c = a ** n`
```python
def __pow__(self, other):
    out = Value(self.data ** other, (self,), f'**{other}')

    def _backward():
        # d(x^n)/dx = n × x^(n-1)
        self.grad += other * (self.data ** (other - 1)) * out.grad

    out._backward = _backward
    return out
```

---

### Tanh (Activation Function)
```python
def tanh(self):
    import math
    t   = math.tanh(self.data)
    out = Value(t, (self,), 'tanh')

    def _backward():
        # d(tanh(x))/dx = 1 - tanh²(x)
        self.grad += (1 - t**2) * out.grad

    out._backward = _backward
    return out
```

**Why tanh was used in Ep1:**
```
tanh output range: -1 to +1 (symmetric around 0)
Works well for small networks like Karpathy's example
Karpathy used it because it's mathematically clean for teaching
```

---

### Division (derived from mul + pow)
```python
def __truediv__(self, other):
    return self * other**-1   # a/b = a × b^(-1)
    # Reuses existing __mul__ and __pow__ — no new backward needed!
```

---

### Subtraction (derived from add + neg)
```python
def __neg__(self):
    return self * -1

def __sub__(self, other):
    return self + (-other)   # a - b = a + (-b)
    # Reuses existing operations!
```

---

## 6. Activation Functions — Comparison

The MOST important section for interviews.

### tanh (used in Ep1)
```python
import math
def tanh(x): return math.tanh(x)
# d(tanh)/dx = 1 - tanh²(x)
```
```
Output range:    -1 to +1
Gradient range:  0 to 1
Zero-centered:   YES (output averages to 0 over batches)
Problems:        Vanishing gradient (when |x| is large, gradient → 0)
Use when:        Hidden layers in small/shallow networks
                 RNN hidden states
```

### Sigmoid
```python
def sigmoid(x): return 1 / (1 + math.exp(-x))
# d(sigmoid)/dx = sigmoid(x) × (1 - sigmoid(x))
```
```
Output range:    0 to 1
Gradient range:  0 to 0.25
Zero-centered:   NO (output always positive → causes zig-zag updates)
Problems:        Vanishing gradient (worse than tanh)
                 Not zero-centered
Use when:        OUTPUT layer for binary classification only
                 Never in hidden layers
```

### ReLU (Rectified Linear Unit)
```python
def relu(x): return max(0, x)
# d(relu)/dx = 1 if x > 0 else 0
```
```
Output range:    0 to infinity
Gradient range:  0 or 1
Zero-centered:   NO
Problems:        "Dying ReLU" — if x < 0 for all inputs, neuron always outputs 0
                               gradient = 0 → neuron never updates → dead
Use when:        Hidden layers in deep CNNs, MLPs
                 Default choice for most modern deep learning
Why better than tanh/sigmoid:
  - Gradient is 1 (not shrinking) → no vanishing gradient for positive inputs
  - Computationally fast (just max(0,x))
```

### Leaky ReLU
```python
def leaky_relu(x, alpha=0.01): return x if x > 0 else alpha * x
# d/dx = 1 if x > 0 else alpha
```
```
Fixes dying ReLU problem — small non-zero gradient for negative inputs
Use when: You're worried about dying ReLU in your architecture
```

### GELU (Gaussian Error Linear Unit)
```python
# Used in BERT, GPT, modern transformers
# Approximate: x × sigmoid(1.702 × x)
```
```
Output:   Smooth version of ReLU
Use when: Transformers (BERT, GPT use this)
Why:      Smoother than ReLU → better gradient flow in very deep networks
```

### Summary Table

| Activation | Range | Vanishing Grad | Dead Neurons | Use Case |
|---|---|---|---|---|
| Sigmoid | (0,1) | ❌ Severe | ❌ No | Output (binary) |
| tanh | (-1,1) | ⚠️ Moderate | ❌ No | Small networks, RNNs |
| ReLU | [0,∞) | ✅ No (pos) | ⚠️ Yes | CNNs, MLPs (default) |
| Leaky ReLU | (-∞,∞) | ✅ No | ✅ No | When ReLU has dying problem |
| GELU | (-∞,∞) | ✅ No | ✅ No | Transformers (BERT, GPT) |

**In Karpathy Ep1 — what if we used ReLU instead of tanh?**
```python
def relu(self):
    out = Value(max(0, self.data), (self,), 'relu')

    def _backward():
        # Gradient is 1 if input > 0, else 0
        self.grad += (self.data > 0) * out.grad

    out._backward = _backward
    return out

# Works exactly the same way. Karpathy used tanh to keep math clean.
# In real networks: use ReLU for hidden layers.
```

---

## 7. The `backward()` Function — Topological Sort

```python
def backward(self):
    topo, visited = [], set()

    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)      # visit children FIRST
            topo.append(v)        # then add self

    build(self)
    self.grad = 1.0               # dLoss/dLoss = 1

    for node in reversed(topo):  # process in REVERSE (output → input)
        node._backward()
```

**Why topological sort?**
```
Before computing gradient of a node →
all nodes that DEPEND on it must have their gradients computed first.

Example: d = a + b;  e = d * c;  f = tanh(e)
Order to compute gradients: f → e → d → (a, b, c)
If we computed gradient of d before e: d.grad = 0 (e hasn't set it yet)
Topological sort ensures correct order automatically.
```

---

## 8. Neuron Class — A Single Artificial Neuron

```python
class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1,1)) for _ in range(n_inputs)]  # weights
        self.b = Value(random.uniform(-1,1))  # bias

    def __call__(self, x):
        # Forward pass: activation(w·x + b)
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]  # all learnable parameters
```

**What a neuron does mathematically:**
```
output = tanh(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
           ↑           ↑                   ↑
       activation   weighted sum          bias
```

**Why random initialisation?**
```
If all weights = 0 → all neurons compute same output → learn same thing
Random init → different neurons → different features → diversity
Range (-1, 1) → small values → prevents activation function saturation initially
```

**Alternative initialisations:**
```python
# Xavier/Glorot (good for tanh, sigmoid):
std = math.sqrt(1 / n_inputs)
self.w = [Value(random.gauss(0, std)) for _ in range(n_inputs)]

# He initialisation (good for ReLU):
std = math.sqrt(2 / n_inputs)
self.w = [Value(random.gauss(0, std)) for _ in range(n_inputs)]
```

---

## 9. Layer Class

```python
class Layer:
    def __init__(self, n_inputs, n_outputs):
        # n_outputs neurons, each taking n_inputs
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        # Each neuron processes same input x
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
```

**What a Layer does:**
```
Input x (n_inputs values)
   ↓ (each neuron processes entire x)
Output (n_outputs values, one per neuron)

Layer(3, 4): 3 inputs → 4 neurons → 4 outputs
```

---

## 10. MLP Class (Multi-Layer Perceptron)

```python
class MLP:
    def __init__(self, n_inputs, layer_sizes):
        sizes = [n_inputs] + layer_sizes
        self.layers = [Layer(sizes[i], sizes[i+1])
                       for i in range(len(layer_sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)   # output of one layer = input of next
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

**Example: MLP(3, [4, 4, 1])**
```
Layer 1: 3 inputs → 4 neurons (4 outputs)
Layer 2: 4 inputs → 4 neurons (4 outputs)
Layer 3: 4 inputs → 1 neuron  (1 output = final prediction)

Total parameters: (3×4 + 4) + (4×4 + 4) + (4×1 + 1) = 16 + 20 + 5 = 41
```

---

## 11. Training Loop — Every Step Explained

```python
for epoch in range(100):

    # STEP 1: FORWARD PASS
    y_pred = [model(x) for x in X_data]
    # Model makes predictions using current weights

    # STEP 2: COMPUTE LOSS
    loss = sum((yp - yt)**2 for yp, yt in zip(y_pred, y_data))
    # MSE Loss: measures how wrong the predictions are
    # Lower loss = better predictions
    # This creates the computation graph

    # STEP 3: ZERO GRADIENTS
    for p in model.parameters():
        p.grad = 0.0
    # WHY: gradients accumulate with +=
    # Without zeroing → old gradients from last epoch add to new ones
    # → wrong direction updates
    # This is exactly what PyTorch's optimizer.zero_grad() does

    # STEP 4: BACKWARD PASS
    loss.backward()
    # Computes dLoss/dw for EVERY weight w in the network
    # Uses topological sort + chain rule automatically
    # This is exactly what PyTorch's loss.backward() does

    # STEP 5: UPDATE WEIGHTS
    for p in model.parameters():
        p.data -= 0.05 * p.grad
    # Gradient descent: move weights in direction that reduces loss
    # 0.05 = learning rate
    # This is exactly what PyTorch's optimizer.step() does
```

**The same 5 steps in PyTorch:**
```python
for epoch in range(100):
    y_pred = model(X)              # Step 1: Forward
    loss   = loss_fn(y_pred, y)    # Step 2: Loss
    optimizer.zero_grad()          # Step 3: Zero grads
    loss.backward()                # Step 4: Backward
    optimizer.step()               # Step 5: Update
```

**They are IDENTICAL — just automated.**

---

## 12. Loss Functions — Alternatives to MSE

**MSE (Mean Squared Error) — used in Ep1:**
```python
loss = sum((yp - yt)**2 for yp, yt in zip(y_pred, y_data))
# Good for: regression (predicting numbers)
# Problem: penalises large errors quadratically (outliers have huge effect)
```

**MAE (Mean Absolute Error):**
```python
loss = sum(abs(yp.data - yt) for yp, yt in zip(y_pred, y_data))
# Good for: regression with outliers
# Problem: gradient is undefined at 0
```

**Binary Cross-Entropy (BCE):**
```python
loss = -sum(yt*log(yp) + (1-yt)*log(1-yp) for yp, yt in zip(y_pred, y_data))
# Good for: binary classification (0 or 1 output)
# Works with sigmoid output layer
```

**In PyTorch:**
```python
nn.MSELoss()           # regression
nn.BCELoss()           # binary classification (with sigmoid)
nn.CrossEntropyLoss()  # multi-class (with softmax, most common)
```

---

## 13. Connection to PyTorch

Every concept in micrograd maps directly to PyTorch:

| Micrograd | PyTorch | What it does |
|---|---|---|
| `Value.data` | `tensor.data` | The actual number |
| `Value.grad` | `tensor.grad` | The gradient |
| `Value._backward` | autograd engine | Chain rule computation |
| `loss.backward()` | `loss.backward()` | Compute all gradients |
| `p.data -= lr * p.grad` | `optimizer.step()` | Update weights |
| `p.grad = 0.0` | `optimizer.zero_grad()` | Clear old gradients |
| `Neuron` | `nn.Linear(1,1)` | Single neuron/layer |
| `Layer` | `nn.Linear(in, out)` | Full layer |
| `MLP` | `nn.Sequential(...)` | Multi-layer network |
| `.parameters()` | `.parameters()` | Same name, same purpose |

---

## 14. Interview Q&A

**Q1: What is backpropagation?**
> Backpropagation is the algorithm for computing gradients of the loss with respect to every weight in the network. It applies the chain rule to propagate gradients backward through the computation graph — from the loss function to the input layer. I implemented this from scratch in micrograd where each operation stores its own backward function that computes local gradients.

**Q2: What is the chain rule?**
> The chain rule states that if y = f(g(x)), then dy/dx = (dy/dg) × (dg/dx). In neural networks, the loss is a composition of many functions. Chain rule lets us compute dLoss/dw for any weight w by multiplying gradients at each step from output to input.

**Q3: Why do we zero gradients before backward?**
> Gradients accumulate using +=. If we don't zero before each backward pass, gradients from the previous batch add to the current batch, giving wrong update directions. In PyTorch this is optimizer.zero_grad(). In micrograd we manually set p.grad = 0.

**Q4: What is vanishing gradient?**
> When gradients become very small as they propagate backward through many layers. Sigmoid and tanh have maximum gradients of 0.25 and 1.0 respectively. Multiplying many values < 1 → exponentially small gradients → early layers learn very slowly. Solved by ReLU (gradient = 1 for positive inputs) and residual connections.

**Q5: Why use ReLU instead of tanh?**
> ReLU has gradient = 1 for all positive inputs, preventing vanishing gradient in deep networks. It's also computationally cheap (just max(0,x)). tanh has gradient between 0-1, which shrinks as it propagates through many layers. In practice: use ReLU for hidden layers in deep networks, tanh for RNNs.

**Q6: What is the difference between tanh and sigmoid?**
> Both are S-shaped curves. Sigmoid outputs 0-1 (used for probability/binary output). tanh outputs -1 to +1 (zero-centered, better for hidden layers). tanh = 2×sigmoid(2x) - 1. tanh has stronger gradients (max 1.0 vs max 0.25 for sigmoid) making it better than sigmoid for hidden layers, though both suffer from vanishing gradient in very deep networks.

**Q7: What is a computation graph?**
> A directed acyclic graph (DAG) where nodes are Values (numbers) and edges represent operations. Each node stores: its value (forward pass), its gradient (filled by backward pass), and a _backward function (chain rule for that operation). The graph is built automatically during the forward pass.

**Q8: Why is the order of backward pass important?**
> We must compute the gradient of a node only AFTER computing gradients of all nodes that depend on it. This is called topological ordering. In micrograd, backward() sorts nodes topologically and processes them in reverse — from output (loss) to inputs (weights). Processing in wrong order gives zero gradients (because dependent nodes haven't set their gradients yet).

**Q9: What happens if learning rate is too high?**
> Weight updates overshoot the minimum. Loss oscillates or diverges (explodes to infinity/NaN). In micrograd: p.data -= lr × p.grad. If lr = 10, each step is 10× too large — the model goes past the minimum and to the other side, then overcorrects, never converging.

**Q10: What is the difference between micrograd and PyTorch?**
> Micrograd is a simplified educational autograd engine. PyTorch uses the same principles but: supports tensors (not just scalars), runs on GPU, has optimized C++/CUDA implementations, includes pre-built layers (nn.Linear, nn.Conv2d), and supports many more operations. The core idea — forward pass builds computation graph, backward pass applies chain rule — is identical.

---

## 15. Quick Reference — What Each File Does

```
ep1_micrograd.ipynb should contain:

Cell 1:  Value class (full implementation with all operations)
Cell 2:  Forward pass test (a*b + c → verify values)
Cell 3:  Manual backward test (verify gradients match numerical)
Cell 4:  Numerical gradient verification
Cell 5:  Neuron class
Cell 6:  Layer class
Cell 7:  MLP class
Cell 8:  Training loop on tiny dataset
Cell 9:  Loss curve plot (loss vs epoch)
Cell 10: Verify final predictions match targets
```
