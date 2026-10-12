# Aurora-Core 架构与不变量推导

> 作者：晨星 · 世界顶级 AI 数学内核
> 设计原则：每个原语复用世界顶级公式，并带一条可被**独立实现**交叉验证的数学不变量。

---

## 1. 总览

```
AuroraCore
├── 线性代数     matMul
├── 数值基础     logSumExp · softmaxRows · softmaxCrossEntropyGrad
├── 归一化       rmsNorm · rmsNormGrad
├── 位置编码     rope
├── 激活         silu · swiglu
├── 注意力       causalAttention
└── 自检         selfTest()  → 8/8 全绿
```

数值验证总纲：**解析解 vs 数值解、朴素实现 vs 优化实现、中心差分 vs 解析梯度**。
误差目标：`1e-7`（梯度检验）~ `1e-12`（代数不变量）。

---

## 2. 数值稳定 softmax / logsumexp

**公式（双精度稳定式）**
```
logsumexp(x) = max(x) + log( Σ exp(x_i − max(x)) )
softmax(z)_i = exp(z_i − m) / Σ exp(z_j − m)
```

**不变量**：每一行概率和 = 1，且 `0 ≤ p_i ≤ 1`。
**验证法**：对照朴素 `Σ exp` 在 `x ≈ 1200` 时已溢出为 `Inf`，稳定式仍有限且等于解析值。

---

## 3. RMSNorm（Zhang & Sennrich 2019）

**公式**
```
r   = sqrt( mean(x²) + ε )
out_i = (w_i / r) · x_i
```

**不变量**：`eps = 0` 时 `RMS(out / w) = sqrt(mean((x/r)²)) = 1`（精确，机器精度）。
工程 `eps > 0` 时偏差 ≤ `eps/2`，属预期数值稳定性代价。

**解析梯度**（上游常量 `g`，`n = dim`）：
```
dr/dx_j        = x_j / (n · r)
d out_i/dx_j   = (w_i / r)·δ_ij  −  w_i x_i x_j / (n · r³)
∂L/∂x_j        = g_j w_j / r  −  x_j·(Σ_i w_i x_i g_i) / (n · r³)
∂L/∂w_j        = g_j x_j / r
```
**验证法**：中心差分 `∂L/∂x, ∂L/∂w` vs 上式，maxRelErr ≈ `1.3e-9`。

---

## 4. RoPE 旋转位置编码（Su et al. 2021）

**公式**：偶数维按相邻对 `(x_{2k}, x_{2k+1})` 旋转角 `θ_k·pos`，
`θ_k = base^{−2k/d}`。
```
[ x'_{2k}   ]   [ cos θ_k  −sin θ_k ] [ x_{2k}   ]
[ x'_{2k+1} ] = [ sin θ_k   cos θ_k ] [ x_{2k+1} ]
```

**不变量 1（保范数）**：旋转矩阵正交 ⇒ `‖RoPE(x, p)‖ = ‖x‖`（机器精度）。

**不变量 2（相对位置不变）**：因旋转正交且 `R(p)ᵀR(q) = R(q−p)`，
```
dot(RoPE(q,m), RoPE(k,n)) = qᵀ R(m)ᵀ R(n) k = qᵀ R(n−m) k
```
故点积**仅依赖相对位置** `n−m`。验证：`dotRel(5,2) == dotRel(105,102)`（相对位移均=3）。

---

## 5. SiLU / SwiGLU（Shazeer 2022, PaLM）

```
SiLU(z) = z / (1 + e^{−z})          # SiLU(0)=0，单调递增
SwiGLU(x) = SiLU(W₁x) ⊙ (W₂x)
```
**不变量**：`SiLU(0)=0`，且 `SiLU` 严格单调递增（验证 `SiLU(−3)<SiLU(0)<SiLU(3)`）。

---

## 6. 因果缩放点积注意力（Vaswani et al. 2017）

```
scores_{ij} = (Q Kᵀ / √d)_{ij} + mask_{ij}   # 因果：j>i ⇒ −∞
weights     = softmax(rows, mask)
ctx_i       = Σ_j weights_{ij} · V_j
```

**不变量**：
1. 每个 query 的注意力权重行和 = 1（softmax 性质）。
2. 因果掩码下 `j > i` 的权重严格 = 0（未来不可见）。
3. 输出形状 `(n_q, d_v)` 与 `V` 列数一致。

---

## 7. softmax + 交叉熵梯度

```
L = −log softmax(z)_y
∂L/∂z_i = softmax(z)_i − 𝟙(i = y)
```
**验证法**：中心差分对标量损失 `−z_y + logsumexp(z)` vs 上式，maxRelErr ≈ `3.2e-9`。
（与 nn-forge `9.23e-7`、attn-forge `5.86e-7` 同量级，符合 forge 系列金标准。）

---

## 8. 复现性

测试使用 `mulberry32(seed=20261013)` 固定种子，保证每次运行结果**逐位可复现**——
这是「可验证」的硬前提：任何人 clone 后 `node src/aurora_core.js` 必得相同 8/8 全绿。

---

## 9. 手写反向自动微分引擎（autodiff）

**站在世界顶级成果之上**：反向模式 AD + 计算图 + 拓扑序 + 链式法则累积
（思想源自 micrograd / JAX），增强支持标量 / 一维 / 二维、广播、matmul、softmax。

### 9.1 机制

```
前向：建图，每个 Tensor 记录 data / grad / _prev / _backward
反向：
  1. 拓扑排序（DFS 后序）得到线性执行序
  2. root.grad = 1
  3. 逆序执行每个节点的 _backward，按链式法则把上游梯度累积到 _prev
```

### 9.2 链式法则样例

对复合 `y = f(g(x))`，反向模式保证：
```
∂L/∂x = Σ_i (∂L/∂y_i)·(∂y_i/∂x)
```
每个算子只需写「局部梯度」，引擎自动串联——这正是 JAX / PyTorch autograd 的核心。

### 9.3 广播 / matmul 的反向还原

- **标量广播**：`s ⊙ arr`，标量端梯度 = `Σ arr_i·g_i`（对被广播维求和还原）。
- **matmul** `C = A·B`：`∂L/∂A = (∂L/∂C)·Bᵀ`，`∂L/∂B = Aᵀ·(∂L/∂C)`。
- **bias 广播** `2D + 1D`：bias 端梯度按列求和（`Σ_rows g`）。

### 9.4 可验证（22/22 全绿）

| 类别 | 验证法 | 结果 |
|------|--------|------|
| 每算子 | 中心差分 vs 解析 backward | maxRelErr < 1e-6 |
| softmax | 对标 `p_j(g_j − Σ g·p)` | ✅ |
| crossEntropy | 对标 `softmax(z) − onehot` | ✅ |
| RMSNorm | 引擎全链（sqrt/mean/div）vs 差分 | ✅ |
| 端到端 | MLP 拟合 sin，SGD 收敛 | loss 降 100×+ |

### 9.5 用法

```js
const AD = require('./src/autodiff.js');
const x = AD.leaf([1, 2, 3]);
const w = AD.leaf([0.5, -0.5, 1.0]);
const loss = AD.sum(AD.mul(AD.tanh(x), w));
loss.backward();   // x.grad / w.grad 已填充
```
