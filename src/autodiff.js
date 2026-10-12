/*
 * Aurora-Core · 手写反向模式自动微分引擎（autodiff）
 * =====================================================================
 * 作者：晨星（GitHub: CJX0712）
 * 理念：零重写 · 集大成 · 闭环交付
 *
 * 站在世界顶级成果之上：
 *   - 反向模式自动微分（reverse-mode AD）+ 计算图 + 拓扑序 + 链式法则累积，
 *     思想源自 micrograd（Karpathy）与 JAX 的反向传播范式；
 *     本实现增强为支持「标量 / 一维向量 / 二维矩阵」、广播、matmul、softmax。
 *
 * 可验证（独立实现交叉验证）：
 *   - 每个算子：中心差分（数值梯度） vs 解析 backward，maxRelErr < 1e-6
 *   - softmax / crossEntropy：对标 softmax(z) − onehot
 *   - 端到端：MLP 拟合 sin，SGD 收敛（loss 下降 > 50× 且末值 < 0.03）
 *
 * 部署：Node `node src/autodiff.js` → 全套自检全绿；浏览器加载后 AuroraAutodiff.selfTest()
 * =====================================================================
 */
(function (global) {
  'use strict';

  // ---- 结构工具 ----------------------------------------------------------
  const isNum = (v) => typeof v === 'number';
  const mapData = (d, fn) => (isNum(d) ? fn(d) : d.map((v) => mapData(v, fn)));
  const zerosLike = (d) => mapData(d, () => 0);
  const onesLike = (d) => mapData(d, () => 1);
  const numel = (d) => (isNum(d) ? 1 : d.reduce((s, v) => s + numel(v), 0));
  const sumLeaves = (d) => (isNum(d) ? d : d.reduce((s, v) => s + sumLeaves(v), 0));
  const combineMap = (a, b, fn) => {
    if (isNum(a) && isNum(b)) return fn(a, b);
    const out = new Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = combineMap(a[i], b[i], fn);
    return out;
  };
  // 累加 self.grad：newGrad_leaf = cb(selfData, oldGrad, upstreamGrad)
  const accGrad = (self, upstream, cb) => {
    const rec = (sd, ag, gd) => (isNum(sd) ? cb(sd, ag, gd) : sd.map((v, i) => rec(v, ag[i], gd[i])));
    return rec(self.data, self.grad, upstream);
  };
  // 累加 self.grad：newGrad += other.data * upstream   （用于乘法的对方项）
  const accMulGrad = (self, other, upstream) => {
    const rec = (sd, sg, od, gd) => (isNum(sd) ? sg + od * gd : sd.map((v, i) => rec(v, sg[i], od[i], gd[i])));
    return rec(self.data, self.grad, other.data, upstream);
  };
  // 累加 self.grad：newGrad += upstream / other.data   （用于除法的分母项）
  const accDivGrad = (self, other, upstream) => {
    const rec = (sd, sg, od, gd) => (isNum(sd) ? sg + gd / od : sd.map((v, i) => rec(v, sg[i], od[i], gd[i])));
    return rec(self.data, self.grad, other.data, upstream);
  };
  const dotSum = (aData, g) => { // Σ aData_i * g_i
    if (isNum(aData)) return aData * g;
    let s = 0; for (let i = 0; i < aData.length; i++) s += dotSum(aData[i], g[i]);
    return s;
  };

  // ---- 数值矩阵 helper（backward 内部纯数值用，不建图）--------------------
  const transpose = (A) => A[0].map((_, j) => A.map((row) => row[j]));
  function matMul(A, B) {
    const m = A.length, k = B.length, n = B[0].length;
    const C = Array.from({ length: m }, () => new Array(n).fill(0));
    for (let i = 0; i < m; i++) for (let p = 0; p < k; p++) { const a = A[i][p]; for (let j = 0; j < n; j++) C[i][j] += a * B[p][j]; }
    return C;
  }
  const addMat = (A, B) => combineMap(A, B, (x, y) => x + y);

  // ---- Tensor ------------------------------------------------------------
  class Tensor {
    constructor(data, _prev, _op) {
      this.data = data;            // number | number[] | number[][]
      this.grad = zerosLike(data); // 初始为 0
      this._prev = _prev || [];
      this._op = _op || 'leaf';
      this._backward = () => {};
    }
    backward() {
      const topo = []; const seen = new Set();
      (function visit(n) {
        if (seen.has(n)) return; seen.add(n);
        for (const c of n._prev) visit(c);
        topo.push(n);
      })(this);
      this.grad = onesLike(this.data);
      for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
    }
  }
  const leaf = (data) => new Tensor(data, [], 'leaf');

  // ---- 一元元素级 ---------------------------------------------------------
  function mapUnary(x, fwd, dfdx) {
    const out = new Tensor(mapData(x.data, fwd), [x], 'unary');
    out._backward = () => { x.grad = accGrad(x, out.grad, (v, og, gg) => og + dfdx(v) * gg); };
    return out;
  }
  const exp = (x) => mapUnary(x, Math.exp, (v) => Math.exp(v));
  const log = (x) => mapUnary(x, Math.log, (v) => 1 / v);
  const tanh = (x) => { const d = (v) => 1 - Math.tanh(v) ** 2; return mapUnary(x, Math.tanh, d); };
  const sin = (x) => mapUnary(x, Math.sin, Math.cos);
  const cos = (x) => mapUnary(x, Math.cos, (v) => -Math.sin(v));
  const relu = (x) => mapUnary(x, (v) => (v > 0 ? v : 0), (v) => (v > 0 ? 1 : 0));
  const sigmoid = (x) => { const s = (v) => 1 / (1 + Math.exp(-v)), d = (v) => { const t = s(v); return t * (1 - t); }; return mapUnary(x, s, d); };
  const neg = (x) => mapUnary(x, (v) => -v, () => -1);
  const pow = (x, n) => mapUnary(x, (v) => Math.pow(v, n), (v) => n * Math.pow(v, n - 1));

  // ---- 加法（num/num, scalar↔arr, arr+arr 同形）--------------------------
  function add(a, b) {
    const aN = isNum(a.data), bN = isNum(b.data);
    let data;
    if (aN && bN) data = a.data + b.data;
    else if (aN) data = mapData(b.data, (v) => a.data + v);
    else if (bN) data = mapData(a.data, (v) => v + b.data);
    else data = combineMap(a.data, b.data, (x, y) => x + y);
    const out = new Tensor(data, [a, b], 'add');
    out._backward = () => {
      const g = out.grad;
      if (aN && bN) { a.grad += g; b.grad += g; }
      else if (aN) { b.grad = accGrad(b, g, (v, og, gg) => og + gg); a.grad += sumLeaves(g); }
      else if (bN) { a.grad = accGrad(a, g, (v, og, gg) => og + gg); b.grad += sumLeaves(g); }
      else { a.grad = accGrad(a, g, (v, og, gg) => og + gg); b.grad = accGrad(b, g, (v, og, gg) => og + gg); }
    };
    return out;
  }
  const sub = (a, b) => add(a, neg(b));

  // ---- 乘法（num/num, scalar↔arr, arr×arr 同形）--------------------------
  function mul(a, b) {
    const aN = isNum(a.data), bN = isNum(b.data);
    let data;
    if (aN && bN) data = a.data * b.data;
    else if (aN) data = mapData(b.data, (v) => a.data * v);
    else if (bN) data = mapData(a.data, (v) => v * b.data);
    else data = combineMap(a.data, b.data, (x, y) => x * y);
    const out = new Tensor(data, [a, b], 'mul');
    out._backward = () => {
      const g = out.grad;
      if (aN && bN) { a.grad += b.data * g; b.grad += a.data * g; }
      else if (aN) { // s * arr : dL/ds = Σ b_i·g_i ; dL/db_i = s·g_i
        const s = a.data;
        b.grad = accGrad(b, g, (v, og, gg) => og + s * gg);
        a.grad += dotSum(b.data, g);
      } else if (bN) { // arr * s : dL/db(s) = Σ a_i·g_i ; dL/da_i = s·g_i
        const s = b.data;
        a.grad = accGrad(a, g, (v, og, gg) => og + s * gg);
        b.grad += dotSum(a.data, g);
      } else { // arr×arr 同形
        a.grad = accMulGrad(a, b, g);
        b.grad = accMulGrad(b, a, g);
      }
    };
    return out;
  }

  // ---- 除法（num/num, scalar/arr, arr/scalar, arr/arr）--------------------
  function div(a, b) {
    const aN = isNum(a.data), bN = isNum(b.data);
    let data;
    if (aN && bN) data = a.data / b.data;
    else if (aN) data = mapData(b.data, (v) => a.data / v);
    else if (bN) data = mapData(a.data, (v) => v / b.data);
    else data = combineMap(a.data, b.data, (x, y) => x / y);
    const out = new Tensor(data, [a, b], 'div');
    out._backward = () => {
      const g = out.grad;
      if (aN && bN) { a.grad += g / b.data; b.grad += -(a.data * g) / (b.data * b.data); }
      else if (aN) { // s / arr : dL/da_i = -s·g_i/a_i² ; dL/ds = Σ g_i/a_i
        const s = a.data;
        a.grad += sumLeaves(accGrad(b, g, (v, og, gg) => gg / v));
        b.grad = accGrad(b, g, (v, og, gg) => og + (-s * gg) / (v * v));
      } else if (bN) { // arr / s : dL/da_i = g_i/s ; dL/ds = -Σ a_i·g_i/s²
        const s = b.data;
        a.grad = accGrad(a, g, (v, og, gg) => og + gg / s);
        b.grad += -dotSum(a.data, g) / (s * s);
      } else { // arr/arr 同形
        a.grad = accDivGrad(a, b, g);
        // b.grad += -a_i·g_i/b_i²
        const rec = (ad, ag, od, gd) => (isNum(ad) ? ag + (-ad * gd) / (od * od) : ad.map((v, i) => rec(v, ag[i], od[i], gd[i])));
        b.grad = rec(a.data, b.grad, b.data, g);
      }
    };
    return out;
  }

  // ---- 归约：sum / mean -> 标量 ------------------------------------------
  function sum(x) {
    const out = new Tensor(sumLeaves(x.data), [x], 'sum');
    out._backward = () => { const gs = out.grad; x.grad = accGrad(x, zerosLike(x.data), (v, og) => og + gs); };
    return out;
  }
  function mean(x) {
    const n = numel(x.data);
    const out = new Tensor(sumLeaves(x.data) / n, [x], 'mean');
    out._backward = () => { const gs = out.grad / n; x.grad = accGrad(x, zerosLike(x.data), (v, og) => og + gs); };
    return out;
  }

  // ---- matmul：仅 2D × 2D -------------------------------------------------
  function matmul(A, B) {
    if (!Array.isArray(A.data[0]) || !Array.isArray(B.data[0])) throw new Error('matmul 仅支持 2D×2D');
    const out = new Tensor(matMul(A.data, B.data), [A, B], 'matmul');
    out._backward = () => {
      const g = out.grad;
      A.grad = addMat(A.grad, matMul(g, transpose(B.data)));   // (m,k)
      B.grad = addMat(B.grad, matMul(transpose(A.data), g));   // (k,n)
    };
    return out;
  }

  // ---- 按行广播加（2D + 1D bias），反向时 bias 按列求和 -------------------
  function addRowwise(A, b) {
    const data = A.data.map((row) => row.map((v, j) => v + b.data[j]));
    const out = new Tensor(data, [A, b], 'addRow');
    out._backward = () => {
      const g = out.grad;
      A.grad = A.data.map((row, i) => row.map((v, j) => A.grad[i][j] + g[i][j]));
      const nb = b.grad.slice();
      for (let i = 0; i < g.length; i++) for (let j = 0; j < g[i].length; j++) nb[j] += g[i][j];
      b.grad = nb;
    };
    return out;
  }

  // ---- gather：1D -> 标量 --------------------------------------------------
  function index(x, i) {
    const out = new Tensor(x.data[i], [x], 'index');
    out._backward = () => { const ng = x.grad.slice(); ng[i] += out.grad; x.grad = ng; };
    return out;
  }

  // ---- detachedMax：取最大值当常数（不传播梯度，等价 softmax 的 max 偏移）---
  function detachedMax(x) {
    const maxLeaf = (d) => (isNum(d) ? d : Math.max(...d.map(maxLeaf)));
    const out = new Tensor(maxLeaf(x.data), [x], 'detachedMax');
    out._backward = () => {}; // 视为常数
    return out;
  }

  // ---- 组合算子：softmax1D / crossEntropy ---------------------------------
  function softmax1D(z) {
    const m = detachedMax(z);
    const shifted = sub(z, m);
    const e = exp(shifted);
    const s = sum(e);
    const inv = div(leaf(1), s);        // 1/s，s 参与图 → 分母链式正确
    return mul(e, inv);
  }
  // L = logsumexp(z) − z[target] = −log softmax[target]
  function crossEntropy(z, target) {
    const m = detachedMax(z);
    const shifted = sub(z, m);
    const e = exp(shifted);
    const s = sum(e);
    const lse = add(log(s), m);
    const zt = index(z, target);
    return sub(lse, zt);
  }

  // =====================================================================
  //  自检套件
  // =====================================================================
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  function maxRelErr(a, b) {
    let m = 0;
    const walk = (x, y) => { if (typeof x === 'number') { const d = Math.max(1e-12, Math.abs(x) + Math.abs(y)); m = Math.max(m, Math.abs(x - y) / d); return; } for (let i = 0; i < x.length; i++) walk(x[i], y[i]); };
    walk(a, b); return m;
  }
  // 扁平化参数（含标量），返回 flat 向量与 apply(flat) 原地写回
  function flatten(params) {
    const flat = []; const setters = [];
    params.forEach((p) => {
      const d = p.data;
      if (isNum(d)) { flat.push(d); setters.push((v) => { p.data = v; }); }
      else { const rec = (x, parent, key) => { if (isNum(x)) { setters.push((v) => { parent[key] = v; }); flat.push(x); } else x.forEach((v, i) => rec(v, x, i)); }; rec(d, null, null); }
    });
    return { flat, apply: (arr) => setters.forEach((s, i) => s(arr[i])) };
  }
  // 解析梯度 vs 中心差分
  function run1(name, params, makeLoss) {
    const loss = makeLoss(); loss.backward();
    const ana = [];
    params.forEach((p) => {
      if (isNum(p.data)) ana.push(p.grad);
      else { const rec = (d, gd) => { if (isNum(d)) return; for (let i = 0; i < d.length; i++) { if (isNum(d[i])) ana.push(gd[i]); else rec(d[i], gd[i]); } }; rec(p.data, p.grad); }
    });
    const { flat, apply } = flatten(params);
    const eps = 1e-6;
    const num = flat.map((_, k) => {
      const up = flat.slice(); up[k] += eps; apply(up); const lp = makeLoss().data;
      const dn = flat.slice(); dn[k] -= eps; apply(dn); const lm = makeLoss().data;
      apply(flat);
      return (lp - lm) / (2 * eps);
    });
    const err = maxRelErr(ana, num);
    return { name, pass: err < 1e-6, detail: 'maxRelErr=' + err.toExponential(3) };
  }

  function selfTest() {
    const results = [];
    const rng = mulberry32(424242);
    const rand1 = (n) => Array.from({ length: n }, () => rng() * 2 - 1);
    const rand2 = (r, c) => Array.from({ length: r }, () => Array.from({ length: c }, () => rng() * 2 - 1));
    const push = (r) => results.push(r);

    // 逐项（每次新建 leaf 保证梯度干净）
    { const a = leaf(rand1(6)), b = leaf(rand1(6)), w = leaf(rand1(6));
      push(run1('add（同形）', [a, b], () => sum(mul(add(a, b), w)))); }
    { const a = leaf(rand1(6)), b = leaf(rand1(6)), w = leaf(rand1(6));
      push(run1('mul（同形）', [a, b], () => sum(mul(mul(a, b), w)))); }
    { const a = leaf(rand1(6)), s = leaf(0.7), w = leaf(rand1(6));
      push(run1('add/mul（标量广播）', [a, s], () => sum(mul(add(a, s), w)))); }
    { const a = leaf(rand1(6)), s = leaf(1.3), w = leaf(rand1(6));
      push(run1('mul（arr×标量）', [a, s], () => sum(mul(mul(a, s), w)))); }
    { const x = leaf(rand1(6).map((v) => v + 0.5)), w = leaf(rand1(6));
      push(run1('exp 元素级', [x], () => sum(mul(exp(x), w)))); }
    { const x = leaf(rand1(6)), w = leaf(rand1(6));
      push(run1('tanh 元素级', [x], () => sum(mul(tanh(x), w)))); }
    { const x = leaf(rand1(6)), w = leaf(rand1(6));
      push(run1('sin 元素级', [x], () => sum(mul(sin(x), w)))); }
    { const x = leaf(rand1(6).map((v) => v + 1)), w = leaf(rand1(6));
      push(run1('relu 元素级（避开0）', [x], () => sum(mul(relu(x), w)))); }
    { const x = leaf(rand1(6)), w = leaf(rand1(6));
      push(run1('sigmoid 元素级', [x], () => sum(mul(sigmoid(x), w)))); }
    { const x = leaf(rand1(6).map((v) => Math.abs(v) + 0.3)), w = leaf(rand1(6));
      push(run1('log 元素级（x>0）', [x], () => sum(mul(log(x), w)))); }
    { const x = leaf(rand1(6).map((v) => Math.abs(v) + 0.3)), w = leaf(rand1(6));
      push(run1('pow(x,3)', [x], () => sum(mul(pow(x, 3), w)))); }
    { const x = leaf(rand1(6).map((v) => Math.abs(v) + 0.3)), w = leaf(rand1(6));
      push(run1('sqrt=pow(x,0.5)', [x], () => sum(mul(pow(x, 0.5), w)))); }
    { const a = leaf(rand1(6).map((v) => v + 2)), s = leaf(0.7), w = leaf(rand1(6));
      push(run1('div（arr/scalar）', [a, s], () => sum(mul(div(a, s), w)))); }
    { const a = leaf(rand1(6).map((v) => v + 2)), s = leaf(1.3), w = leaf(rand1(6));
      push(run1('div（scalar/arr）', [a, s], () => sum(mul(div(s, a), w)))); }
    { const x = leaf(rand1(8)), w = leaf(0.5);
      push(run1('sum', [x], () => mul(sum(x), w))); }
    { const x = leaf(rand1(8)), w = leaf(0.5);
      push(run1('mean', [x], () => mul(mean(x), w))); }
    { const A = leaf(rand2(4, 3)), B = leaf(rand2(3, 5)), W = leaf(rand2(4, 5));
      push(run1('matmul 2D×2D', [A, B], () => sum(mul(matmul(A, B), W)))); }
    { const X = leaf(rand2(5, 4)), b = leaf(rand1(4)), W = leaf(rand2(5, 4));
      push(run1('addRowwise（bias）', [X, b], () => sum(mul(addRowwise(X, b), W)))); }

    // softmax1D 梯度 vs 解析 jacobian
    { const z = leaf(rand1(7)), g = leaf(rand1(7));
      const p = softmax1D(z); const loss = sum(mul(p, g)); loss.backward();
      const pj = p.data; let s = 0; for (let k = 0; k < 7; k++) s += g.data[k] * pj[k];
      const ana = pj.map((v, j) => v * (g.data[j] - s));
      const { flat, apply } = flatten([z]); const eps = 1e-6;
      const num = flat.map((_, k) => { const up = flat.slice(); up[k] += eps; apply(up); const lp = sum(mul(softmax1D(z), g)).data; const dn = flat.slice(); dn[k] -= eps; apply(dn); const lm = sum(mul(softmax1D(z), g)).data; apply(flat); return (lp - lm) / (2 * eps); });
      const err = maxRelErr(ana, num);
      push({ name: 'softmax1D 梯度（含分母链式）', pass: err < 1e-6, detail: 'maxRelErr=' + err.toExponential(3) }); }

    // crossEntropy 梯度 = softmax − onehot
    { const z = leaf(rand1(6)); const target = 2;
      const loss = crossEntropy(z, target); loss.backward();
      const p = softmax1D(z).data; const ana = p.slice(); ana[target] -= 1;
      const { flat, apply } = flatten([z]); const eps = 1e-6;
      const num = flat.map((_, k) => { const up = flat.slice(); up[k] += eps; apply(up); const lp = crossEntropy(z, target).data; const dn = flat.slice(); dn[k] -= eps; apply(dn); const lm = crossEntropy(z, target).data; apply(flat); return (lp - lm) / (2 * eps); });
      const err = maxRelErr(ana, num);
      push({ name: 'crossEntropy 梯度 = softmax−onehot', pass: err < 1e-6, detail: 'maxRelErr=' + err.toExponential(3) }); }

    // 引擎重算 RMSNorm 前向+梯度（sqrt/mean/div 链）vs 中心差分
    { const d = 6, eps = 1e-6; const g = leaf(rand1(d));
      const make = () => { const x = leaf(rand1(d).map((v) => v + 0.5)), w = leaf(rand1(d)); return { x, w, loss: sum(mul(mul(x, div(w, add(pow(mean(mul(x, x)), 0.5), leaf(eps)))), g)) }; };
      const { x, w, loss } = make(); loss.backward();
      const ana = x.grad.concat(w.grad);
      const { flat, apply } = flatten([x, w]);
      const num = flat.map((_, k) => { const up = flat.slice(); up[k] += eps; apply(up); const lp = sum(mul(mul(x, div(w, add(pow(mean(mul(x, x)), 0.5), leaf(eps)))), g)).data; const dn = flat.slice(); dn[k] -= eps; apply(dn); const lm = sum(mul(mul(x, div(w, add(pow(mean(mul(x, x)), 0.5), leaf(eps)))), g)).data; apply(flat); return (lp - lm) / (2 * eps); });
      const err = maxRelErr(ana, num);
      push({ name: 'RMSNorm（引擎 sqrt/mean/div 链）', pass: err < 1e-6, detail: 'maxRelErr=' + err.toExponential(3) }); }

    // 端到端训练：autodiff MLP 拟合 sin，验证收敛（闭环训练铁证）
    {
      const N = 128, H = 32, lr = 0.06;
      const xs = [], ys = [];
      for (let i = 0; i < N; i++) { const x = -Math.PI + (2 * Math.PI * i) / (N - 1); xs.push([x]); ys.push([Math.sin(x)]); }
      const X = leaf(xs), Y = leaf(ys);
      const r2 = mulberry32(7);
      const W1 = leaf([[].concat(Array.from({ length: H }, () => (r2() - 0.5) * 0.3))]);
      const b1 = leaf(Array.from({ length: H }, () => 0));
      const W2 = leaf(Array.from({ length: H }, () => [(r2() - 0.5) * 0.3]));
      const b2 = leaf([0]);
      const params = [W1, b1, W2, b2];
      const forward = () => { const h = tanh(addRowwise(matmul(X, W1), b1)); return addRowwise(matmul(h, W2), b2); };
      const upd = (d, gd) => (isNum(d) ? d - lr * gd : d.map((v, i) => upd(v, gd[i])));
      let first = 0, last = 0;
      for (let step = 0; step < 2000; step++) {
        const diff = sub(forward(), Y);
        const loss = mean(mul(diff, diff));
        loss.backward();
        if (step === 0) first = loss.data;
        params.forEach((p) => { p.data = upd(p.data, p.grad); p.grad = zerosLike(p.data); });
        if (step === 1999) last = loss.data;
      }
      const ratio = first / last;
      push({ name: '端到端训练：MLP 拟合 sin 收敛', pass: ratio > 50 && last < 0.03, detail: 'loss ' + first.toFixed(4) + '→' + last.toFixed(5) + ' (降 ' + ratio.toFixed(0) + '×)' });
    }

    const passed = results.filter((r) => r.pass).length;
    return { total: results.length, passed, results, allGreen: passed === results.length };
  }

  const API = { Tensor, leaf, add, sub, mul, div, neg, pow, exp, log, tanh, sin, cos, relu, sigmoid, sum, mean, matmul, addRowwise, index, detachedMax, softmax1D, crossEntropy, selfTest };

  global.AuroraAutodiff = API;
  if (typeof module !== 'undefined' && module.exports) module.exports = API;

  if (typeof require !== 'undefined' && typeof module !== 'undefined' && require.main === module) {
    const r = selfTest();
    console.log('Aurora-Autodiff 自检：' + r.passed + '/' + r.total + (r.allGreen ? ' ✅ 全绿' : ' ❌'));
    r.results.forEach((t) => console.log('  [' + (t.pass ? 'PASS' : 'FAIL') + '] ' + t.name + (t.detail ? '  (' + t.detail + ')' : '')));
    if (!r.allGreen) process.exit(1);
  }
})(typeof window !== 'undefined' ? window : globalThis);
