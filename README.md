# Aurora-Core · 世界顶级 AI 数学内核

> 作者：**晨星**（GitHub: CJX0712）
> 理念：**零重写 · 集大成 · 闭环交付**
> 形态：纯 JS、零依赖、零构建；Node 与浏览器双跑；**8/8 自检全绿**；每条原语带可被独立实现交叉验证的**数学不变量**。

---

## 这是什么

Aurora-Core 是现代 SOTA 大语言模型（LLaMA / PaLM 系）所依赖的**数学原语内核**——
不是玩具，而是站在世界顶级成果之上的可验证实现。每一个原语都**直接复用**已被工业级
模型验证过的公式，不重造轮子：

| 原语 | 来源（世界顶级成果） | 数学不变量（交叉验证） |
|------|----------------------|------------------------|
| 数值稳定 softmax / logsumexp | 经典数值方法（Gokhale） | 抗溢出；行和 = 1 |
| RMSNorm | Zhang & Sennrich 2019（LLaMA / GPT-NeoX） | 归一化后 RMS = 1（eps=0 精确） |
| RoPE | Su et al. 2021《RoFormer》 | 旋转保范数；点积仅依赖相对位置 |
| SiLU / SwiGLU | Shazeer 2022（PaLM） | SiLU(0)=0、单调递增 |
| 因果缩放点积注意力 | Vaswani et al. 2017（Transformer） | 行和 = 1；未来位置权重 = 0 |
| softmax + 交叉熵梯度 | 解析梯度 | 中心差分 vs 解析，maxRelErr ≈ 1e-7 |

## 为什么是世界顶级

- **集大成**：每个公式都站在已发表、已被顶级模型采用的成果之上，凡有更强现成方案即替换自写。
- **可验证**：不是"看起来对"，而是每条核心性质都有独立实现（朴素三重循环 / 中心差分 /
  解析解）交叉验证，误差到 `1e-7 ~ 1e-12` 量级。
- **零依赖零构建**：单文件 `src/aurora_core.js`，浏览器直接 `<script>` 加载，Node 直接 `require`。
- **可复现**：内置确定性 PRNG（mulberry32，种子固定），测试 100% 可复现。

## 快速开始

```bash
# Node：跑全套自检
node src/aurora_core.js
# => Aurora-Core 自检：8/8 ✅ 全绿

# 浏览器：打开 index.html，一键查看绿/红自检结果
# 或在控制台：AuroraCore.selfTest()
```

```js
const A = require('./src/aurora_core.js');

// RoPE：旋转位置编码（LLaMA 用）
const q = [1, 2, 3, 4];
A.rope(q, 5, 10000);              // => 旋转后向量，范数不变

// RMSNorm（LLaMA 用）
A.rmsNorm([1, 2, 3, 4], [1, 1, 1, 1], 1e-6);

// 因果注意力
A.causalAttention(Q, K, V, true); // => { ctx, weights }
```

## 仓库结构

```
aurora-core/
├── src/aurora_core.js     # 核心内核（零依赖，自带 8/8 自检）
├── index.html             # 浏览器一键自检页（单文件内联）
├── docs/ARCHITECTURE.md   # 架构与不变量推导
├── package.json
└── LICENSE                # MIT
```

## 路线图（集大成，持续超越）

- [ ] 前向+反向全自动微分（手写 autodiff，对标 micrograd / JAX）
- [ ] FlashAttention 风格分块注意力（IO 感知，对标 Dao et al. 2022）
- [ ] KAN 层（Kolmogorov–Arnold，对标 Liu et al. 2024）
- [ ] 完整 Transformer Block 前向+反向（对标 nanoGPT）
- [ ] WASM / SIMD 加速内核

---

**署名：晨星** · 可运行 · 可复现 · 可验证 · 完整开源
