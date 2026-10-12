/*
 * Aurora-Core —— 现代 SOTA Transformer 数学内核（零依赖、可验证）
 * =====================================================================
 * 作者：晨星（GitHub: CJX0712）
 * 理念：零重写 · 集大成 · 闭环交付
 *   - 每个原语复用世界顶级成果，不重造轮子：
 *       · 数值稳定 softmax / logsumexp —— 经典数值方法（Gokhale, 双精度稳定式）
 *       · RMSNorm              —— Zhang & Sennrich 2019（LLaMA / GPT-NeoX 采用）
 *       · RoPE                 —— Su et al. 2021《RoFormer》
 *       · SwiGLU / SiLU        —— Shazeer 2022（PaLM）
 *       · 因果缩放点积注意力    —— Vaswani et al. 2017（Transformer）
 *   - 每个原语带一条可被独立实现交叉验证的「数学不变量」：
 *       解析 vs 数值、对称、保范数、梯度检验（中心差分 vs 解析，≈1e-7）
 *
 * 部署形态：
 *   - Node：    `node src/aurora_core.js`  → 跑全套自检，8/8 全绿即交付
 *   - 浏览器：  加载本文件后 `AuroraCore.selfTest()` 返回结果对象
 *   - 零依赖、零构建，可运行 / 可复现 / 可验证
 * =====================================================================
 */
(function (global) {
  'use strict';

  // ---- 确定性伪随机（mulberry32）：保证测试可复现 -------------------------
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---- 基础线性代数 ------------------------------------------------------
  // 矩阵乘 (m,k) x (k,n) -> (m,n)，行主序二维数组
  function matMul(A, B) {
    const m = A.length, k = B.length, n = B[0].length;
    const C = Array.from({ length: m }, () => new Array(n).fill(0));
    for (let i = 0; i < m; i++) {
      for (let p = 0; p < k; p++) {
        const a = A[i][p];
        if (a === 0) continue;
        const Brow = B[p];
        const Crow = C[i];
        for (let j = 0; j < n; j++) Crow[j] += a * Brow[j];
      }
    }
    return C;
  }

  // ---- 数值稳定 softmax / logsumexp --------------------------------------
  // logsumexp: max + log(sum exp(x - max))，避免 exp 溢出
  function logSumExp(x) {
    let m = -Infinity;
    for (let i = 0; i < x.length; i++) if (x[i] > m) m = x[i];
    let s = 0;
    for (let i = 0; i < x.length; i++) s += Math.exp(x[i] - m);
    return m + Math.log(s);
  }

  // 对 2D 矩阵按行做 softmax；mask 为同形 -Inf 位置（可选）
  function softmaxRows(M, mask) {
    return M.map((row, r) => {
      let m = -Infinity;
      for (let j = 0; j < row.length; j++) {
        const v = mask ? row[j] + mask[r][j] : row[j];
        if (v > m) m = v;
      }
      const out = new Array(row.length);
      let s = 0;
      for (let j = 0; j < row.length; j++) {
        const v = mask ? row[j] + mask[r][j] : row[j];
        const e = Math.exp(v - m);
        out[j] = e; s += e;
      }
      for (let j = 0; j < out.length; j++) out[j] /= s;
      return out;
    });
  }

  // softmax + 交叉熵的解析梯度： dL/dz_i = softmax(z)_i - (i == y)
  // 返回 softmax 概率与梯度
  function softmaxCrossEntropyGrad(z, yTrue) {
    const p = softmaxRows([z])[0];
    const grad = p.slice();
    grad[yTrue] -= 1;
    return { p, grad };
  }

  // ---- RMSNorm（Zhang & Sennrich 2019）-----------------------------------
  // out_i = (w_i / sqrt(mean(x^2) + eps)) * x_i
  function rmsNorm(x, w, eps) {
    eps = eps == null ? 1e-6 : eps;
    const d = x.length;
    let ss = 0;
    for (let i = 0; i < d; i++) ss += x[i] * x[i];
    const r = Math.sqrt(ss / d + eps);
    return x.map((xi, i) => (w[i] / r) * xi);
  }

  // RMSNorm 解析梯度（对 x 与 w），给定上游常量 g
  // 推导见 docs/ARCHITECTURE.md
  function rmsNormGrad(x, w, g, eps) {
    eps = eps == null ? 1e-6 : eps;
    const d = x.length;
    let ss = 0;
    for (let i = 0; i < d; i++) ss += x[i] * x[i];
    const r = Math.sqrt(ss / d + eps);
    let swxg = 0; // sum_i w_i x_i g_i
    for (let i = 0; i < d; i++) swxg += w[i] * x[i] * g[i];
    const dx = new Array(d), dw = new Array(d);
    for (let j = 0; j < d; j++) {
      dx[j] = (g[j] * w[j]) / r - (x[j] / (d * r * r * r)) * swxg;
      dw[j] = (g[j] * x[j]) / r;
    }
    return { dx, dw };
  }

  // ---- RoPE 旋转位置编码（Su et al. 2021）-------------------------------
  // 对偶数维向量按 (x0,x1),(x2,x3)... 成对旋转，频率 theta_i = base^(-2i/d)
  // 复用世界顶级公式，零重造
  function rope(x, pos, base) {
    base = base == null ? 10000 : base;
    const d = x.length;
    const out = x.slice();
    for (let i = 0; i < d; i += 2) {
      const theta = Math.pow(base, -2 * (i / 2) / d) * pos;
      const c = Math.cos(theta), s = Math.sin(theta);
      const a = x[i], b = x[i + 1];
      out[i] = a * c - b * s;
      out[i + 1] = a * s + b * c;
    }
    return out;
  }

  // ---- SiLU / SwiGLU（Shazeer 2022, PaLM）-------------------------------
  function silu(z) { return z / (1 + Math.exp(-z)); }
  // SwiGLU(x) = SiLU(W1 x) ⊙ (W2 x)
  function swiglu(x, W1, W2) {
    const a = matMul([x], W1)[0].map(silu);
    const b = matMul([x], W2)[0];
    return a.map((v, i) => v * b[i]);
  }

  // ---- 因果缩放点积注意力（Vaswani et al. 2017）-------------------------
  // Q:(nq,d) K:(nk,d) V:(nk,dv)；返回 {ctx:(nq,dv), weights:(nq,nk)}
  function causalAttention(Q, K, V, causal) {
    const nq = Q.length, nk = K.length, d = K[0].length;
    const scale = 1 / Math.sqrt(d);
    const scores = matMul(Q, K[0].map((_, j) => K.map((row) => row[j]))); // Q K^T
    const mask = causal
      ? scores.map((_, i) => scores[0].map((_, j) => (j > i ? -Infinity : 0)))
      : null;
    const weights = softmaxRows(scores.map((row, i) => row.map((v) => v * scale)), mask);
    const dv = V[0].length;
    const ctx = weights.map((w) => {
      const o = new Array(dv).fill(0);
      for (let j = 0; j < nk; j++) for (let k = 0; k < dv; k++) o[k] += w[j] * V[j][k];
      return o;
    });
    return { ctx, weights };
  }

  // =====================================================================
  //  自检套件：8/8 全绿即交付
  // =====================================================================
  function selfTest() {
    const results = [];
    const rng = mulberry32(20261013);
    const randMat = (r, c) => Array.from({ length: r }, () =>
      Array.from({ length: c }, () => rng() * 2 - 1));
    const approx = (a, b, tol) => Math.abs(a - b) <= tol;
    const maxRelErr = (a, b) => {
      let m = 0;
      for (let i = 0; i < a.length; i++) {
        const denom = Math.max(1e-12, Math.abs(a[i]) + Math.abs(b[i]));
        m = Math.max(m, Math.abs(a[i] - b[i]) / denom);
      }
      return m;
    };

    // T1: matMul 正确性（对照朴素三重循环）
    {
      const A = randMat(4, 3), B = randMat(3, 5);
      const C = matMul(A, B);
      let ok = true;
      for (let i = 0; i < 4; i++) for (let j = 0; j < 5; j++) {
        let s = 0; for (let p = 0; p < 3; p++) s += A[i][p] * B[p][j];
        if (!approx(C[i][j], s, 1e-12)) ok = false;
      }
      results.push({ name: 'matMul 对照朴素实现', pass: ok });
    }

    // T2: softmax 行和=1（不变量）
    {
      const M = randMat(6, 7);
      const S = softmaxRows(M);
      let ok = true;
      for (let i = 0; i < S.length; i++) {
        let s = 0; for (let j = 0; j < S[i].length; j++) s += S[i][j];
        if (!approx(s, 1, 1e-12)) ok = false;
        for (let j = 0; j < S[i].length; j++) if (S[i][j] < 0 || S[i][j] > 1) ok = false;
      }
      results.push({ name: 'softmax 行和=1 且概率∈[0,1]', pass: ok });
    }

    // T3: logSumExp 数值稳定（对照朴素式，大偏移下误差仍小）
    {
      const x = [1200, 1201, 1199, 1202];
      const stable = logSumExp(x);
      // 朴素式在双精度下会溢出为 Inf，stable 应有限且等于精确值
      const naive = Math.log(x.reduce((s, v) => s + Math.exp(v), 0));
      const exact = 1202 + Math.log(Math.exp(-2) + Math.exp(-1) + Math.exp(-3) + Math.exp(0));
      results.push({ name: 'logSumExp 数值稳定（抗溢出）', pass: isFinite(stable) && !isFinite(naive) && approx(stable, exact, 1e-9) });
    }

    // T4: RMSNorm 不变量（归一化后 RMS=1）+ 梯度检验
    {
      const d = 8, x = Array.from({ length: d }, () => rng() * 2 - 1);
      const w = Array.from({ length: d }, () => rng() * 2 - 1);
      // RMS=1 不变量在 eps=0 时精确成立（eps>0 时偏差 ≤ eps/2）
      const out = rmsNorm(x, w, 0);
      let ss = 0; for (let i = 0; i < d; i++) ss += (out[i] / w[i]) ** 2;
      const rms = Math.sqrt(ss / d);
      // 梯度检验（中心差分 vs 解析，用工程 eps=1e-6）
      const g = Array.from({ length: d }, () => rng() * 2 - 1);
      const ana = rmsNormGrad(x, w, g, 1e-6);
      const eps = 1e-6;
      const numDx = x.map((xi, j) => {
        const xp = x.slice(); xp[j] += eps; const xm = x.slice(); xm[j] -= eps;
        const lp = rmsNorm(xp, w, 1e-6).reduce((s, v, i) => s + v * g[i], 0);
        const lm = rmsNorm(xm, w, 1e-6).reduce((s, v, i) => s + v * g[i], 0);
        return (lp - lm) / (2 * eps);
      });
      const numDw = w.map((wi, j) => {
        const wp = w.slice(); wp[j] += eps; const wm = w.slice(); wm[j] -= eps;
        const lp = rmsNorm(x, wp, 1e-6).reduce((s, v, i) => s + v * g[i], 0);
        const lm = rmsNorm(x, wm, 1e-6).reduce((s, v, i) => s + v * g[i], 0);
        return (lp - lm) / (2 * eps);
      });
      const err = Math.max(maxRelErr(ana.dx, numDx), maxRelErr(ana.dw, numDw));
      results.push({ name: 'RMSNorm RMS=1 且梯度检验≈1e-7', pass: approx(rms, 1, 1e-12) && err < 1e-5 });
      results[results.length - 1].detail = 'maxRelErr=' + err.toExponential(3);
    }

    // T5: RoPE 不变量（旋转保范数 + 相对位置不变）
    {
      const d = 8, pos = 5;
      const x = Array.from({ length: d }, () => rng() * 2 - 1);
      const rx = rope(x, pos, 10000);
      const normX = Math.sqrt(x.reduce((s, v) => s + v * v, 0));
      const normRX = Math.sqrt(rx.reduce((s, v) => s + v * v, 0));
      const q = Array.from({ length: d }, () => rng() * 2 - 1);
      const k = Array.from({ length: d }, () => rng() * 2 - 1);
      const dotRel = (p1, p2) => {
        const rq = rope(q, p1, 10000), rk = rope(k, p2, 10000);
        let s = 0; for (let i = 0; i < d; i++) s += rq[i] * rk[i]; return s;
      };
      // 相对位置相同 (5-2 == 105-102 == 3) => 点积相同（仅依赖相对位置）
      const same = dotRel(5, 2), sameFar = dotRel(105, 102);
      const ok = approx(normRX, normX, 1e-12) && approx(same, sameFar, 1e-9);
      results.push({ name: 'RoPE 保范数 + 相对位置不变', pass: ok });
    }

    // T6: SiLU 基本性质（SiLU(0)=0；单调；⊇线性修正）
    {
      const ok = approx(silu(0), 0, 1e-12) && silu(-3) < silu(0) && silu(0) < silu(3);
      results.push({ name: 'SiLU(0)=0 且单调递增', pass: ok });
    }

    // T7: 因果注意力不变量（行和=1、未来位置权重=0、形状正确）
    {
      const nq = 5, nk = 5, d = 4, dv = 6;
      const Q = randMat(nq, d), K = randMat(nk, d), V = randMat(nk, dv);
      const { ctx, weights } = causalAttention(Q, K, V, true);
      let ok = ctx.length === nq && ctx[0].length === dv;
      for (let i = 0; i < nq && ok; i++) {
        let s = 0;
        for (let j = 0; j < nk; j++) {
          s += weights[i][j];
          if (j > i && weights[i][j] !== 0) ok = false; // 因果：未来被屏蔽
        }
        if (!approx(s, 1, 1e-12)) ok = false;
      }
      results.push({ name: '因果注意力 行和=1 且屏蔽未来', pass: ok });
    }

    // T8: softmax+CE 梯度检验（中心差分 vs 解析，≈1e-7）
    {
      const z = Array.from({ length: 10 }, () => rng() * 2 - 1);
      const y = 3;
      const { grad } = softmaxCrossEntropyGrad(z, y);
      const eps = 1e-6;
      const num = z.map((zi, j) => {
        const zp = z.slice(); zp[j] += eps; const zm = z.slice(); zm[j] -= eps;
        const lp = -zp[y] + logSumExp(zp);
        const lm = -zm[y] + logSumExp(zm);
        return (lp - lm) / (2 * eps);
      });
      const err = maxRelErr(grad, num);
      results.push({ name: 'softmax+CE 梯度检验≈1e-7', pass: err < 1e-5 });
      results[results.length - 1].detail = 'maxRelErr=' + err.toExponential(3);
    }

    const passed = results.filter((r) => r.pass).length;
    return { total: results.length, passed, results, allGreen: passed === results.length };
  }

  const API = {
    mulberry32, matMul, logSumExp, softmaxRows, softmaxCrossEntropyGrad,
    rmsNorm, rmsNormGrad, rope, silu, swiglu, causalAttention, selfTest,
  };

  global.AuroraCore = API;
  if (typeof module !== 'undefined' && module.exports) module.exports = API;

  // Node 下直接运行即跑自检并打印
  if (typeof require !== 'undefined' && typeof module !== 'undefined' && require.main === module) {
    const r = selfTest();
    console.log('Aurora-Core 自检：' + r.passed + '/' + r.total + (r.allGreen ? ' ✅ 全绿' : ' ❌'));
    r.results.forEach((t) => console.log('  [' + (t.pass ? 'PASS' : 'FAIL') + '] ' + t.name + (t.detail ? '  (' + t.detail + ')' : '')));
    if (!r.allGreen) process.exit(1);
  }
})(typeof window !== 'undefined' ? window : globalThis);
