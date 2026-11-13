// The same small GPT as model.py, in plain JavaScript, so the page needs no server and no libraries.
// It remembers its work on the shapes already drawn, so each new suggestion only costs a few steps.

export const PAD = 0, BOS = 1, EOS = 2, LINE = 3, CIRCLE = 4, ARC = 5, POINT = 6, CONSTRUCTION = 4;
export const FIRST_TYPE = 3, LAST_TYPE = 10, COORD0 = 11, BINS = 64, VOCAB = 75, BLOCK = 128, MAX_SHAPES = 16;
const N_COORDS = { 3: 4, 4: 3, 5: 6, 6: 2 };
export const baseType = t => (t > POINT ? t - CONSTRUCTION : t);
export const nCoords = t => N_COORDS[baseType(t)];
export const isConstruction = t => t > POINT;

function halfToFloat(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    out[i] = e === 0 ? s * f * 5.960464477539063e-8 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return out;
}

export function buildModel(manifest, buffer) {
  const all = halfToFloat(new Uint16Array(buffer));
  const w = {};
  for (const t of manifest.tensors) w[t.name] = all.subarray(t.offset, t.offset + t.shape.reduce((a, b) => a * b, 1));
  const model = new Model(manifest.cfg, w);
  model.trainedSteps = manifest.step;
  return model;
}

export async function loadModel(base = '.') {
  const [manifest, buffer] = await Promise.all([
    fetch(`${base}/weights.json`).then(r => r.json()),
    fetch(`${base}/weights.bin`).then(r => r.arrayBuffer()),
  ]);
  return buildModel(manifest, buffer);
}

function linear(W, b, x, nOut, nIn, out) {
  for (let o = 0, k = 0; o < nOut; o++) {
    let s = b[o];
    for (let i = 0; i < nIn; i++, k++) s += W[k] * x[i];
    out[o] = s;
  }
  return out;
}

function layerNorm(g, b, x, out) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - mean) ** 2;
  const inv = 1 / Math.sqrt(v / n + 1e-5);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * g[i] + b[i];
  return out;
}

class Model {
  constructor(cfg, w) {
    this.d = cfg.d; this.layers = cfg.layers; this.heads = cfg.heads; this.w = w;
    this.tmp = new Float32Array(cfg.d); this.qkv = new Float32Array(3 * cfg.d);
    this.att = new Float32Array(cfg.d); this.hid = new Float32Array(4 * cfg.d);
    this.reset();
  }

  reset() { this.tokens = []; this.memory = this.emptyMemory(); this.logits = null; }
  emptyMemory() { return Array.from({ length: this.layers }, () => ({ k: [], v: [] })); }
  cloneMemory(m) { return m.map(l => ({ k: l.k.slice(), v: l.v.slice() })); }

  // Feed one token at position pos; memory grows by one entry per layer. Returns the scores for the next token.
  step(tok, pos, memory) {
    const { d, heads, w } = this, hd = d / heads, scale = 1 / Math.sqrt(hd);
    const x = new Float32Array(d);
    for (let i = 0; i < d; i++) x[i] = w['tok.weight'][tok * d + i] + w['pos.weight'][pos * d + i];
    for (let l = 0; l < this.layers; l++) {
      const p = `blocks.${l}.`, mem = memory[l];
      layerNorm(w[p + 'ln1.weight'], w[p + 'ln1.bias'], x, this.tmp);
      linear(w[p + 'qkv.weight'], w[p + 'qkv.bias'], this.tmp, 3 * d, d, this.qkv);
      mem.k.push(this.qkv.slice(d, 2 * d)); mem.v.push(this.qkv.slice(2 * d));
      const n = mem.k.length, score = new Float32Array(n);
      for (let h = 0; h < heads; h++) {
        const o = h * hd;
        let top = -Infinity;
        for (let t = 0; t < n; t++) {
          const k = mem.k[t];
          let s = 0;
          for (let i = 0; i < hd; i++) s += this.qkv[o + i] * k[o + i];
          score[t] = s * scale;
          if (score[t] > top) top = score[t];
        }
        let sum = 0;
        for (let t = 0; t < n; t++) { score[t] = Math.exp(score[t] - top); sum += score[t]; }
        for (let i = 0; i < hd; i++) this.att[o + i] = 0;
        for (let t = 0; t < n; t++) {
          const a = score[t] / sum, v = mem.v[t];
          for (let i = 0; i < hd; i++) this.att[o + i] += a * v[o + i];
        }
      }
      linear(w[p + 'proj.weight'], w[p + 'proj.bias'], this.att, d, d, this.tmp);
      for (let i = 0; i < d; i++) x[i] += this.tmp[i];
      layerNorm(w[p + 'ln2.weight'], w[p + 'ln2.bias'], x, this.tmp);
      linear(w[p + 'fc1.weight'], w[p + 'fc1.bias'], this.tmp, 4 * d, d, this.hid);
      for (let i = 0; i < 4 * d; i++) {
        const v = this.hid[i];
        this.hid[i] = 0.5 * v * (1 + Math.tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
      }
      linear(w[p + 'fc2.weight'], w[p + 'fc2.bias'], this.hid, d, 4 * d, this.tmp);
      for (let i = 0; i < d; i++) x[i] += this.tmp[i];
    }
    layerNorm(w['ln.weight'], w['ln.bias'], x, this.tmp);
    return linear(w['head.weight'], w['head.bias'], this.tmp, VOCAB, d, new Float32Array(VOCAB));
  }

  // Bring the memory up to date with the sketch so far (re-using everything that has not changed).
  read(tokens) {
    let same = 0;
    while (same < this.tokens.length && same < tokens.length && this.tokens[same] === tokens[same]) same++;
    if (same < this.tokens.length) {
      this.tokens.length = same;
      for (const l of this.memory) { l.k.length = same; l.v.length = same; }
      this.logits = null;
    }
    for (let i = same; i < tokens.length; i++) { this.logits = this.step(tokens[i], i, this.memory); this.tokens.push(tokens[i]); }
    if (!this.logits && tokens.length) {      // sketch got shorter: recompute the last position
      const last = tokens.length - 1;
      this.tokens.length = last;
      for (const l of this.memory) { l.k.length = last; l.v.length = last; }
      this.logits = this.step(tokens[last], last, this.memory); this.tokens.push(tokens[last]);
    }
    return this.logits;
  }

  // Best few guesses for the next shape: [{p, tokens}] best first. tokens [EOS] means "looks finished".
  suggest(tokens, width = 5) {
    const first = this.read(tokens);
    const logSoftmax = (logits, lo, hi, extra) => {
      let top = -Infinity;
      for (let i = lo; i <= hi; i++) top = Math.max(top, logits[i]);
      if (extra !== undefined) top = Math.max(top, logits[extra]);
      let sum = 0;
      for (let i = lo; i <= hi; i++) sum += Math.exp(logits[i] - top);
      if (extra !== undefined) sum += Math.exp(logits[extra] - top);
      const z = top + Math.log(sum), out = [];
      for (let i = lo; i <= hi; i++) out.push([i, logits[i] - z]);
      if (extra !== undefined) out.push([extra, logits[extra] - z]);
      return out;
    };
    const room = tokens.length + 7 <= BLOCK;
    let beams = logSoftmax(first, FIRST_TYPE, LAST_TYPE, EOS)
      .filter(([t]) => t === EOS || room)
      .sort((a, b) => b[1] - a[1]).slice(0, width)
      .map(([t, lp]) => ({ lp, toks: [t], left: t === EOS ? 0 : nCoords(t), memory: null }));
    for (let round = 0; round < 6 && beams.some(b => b.left > 0); round++) {
      const next = [];
      for (const b of beams) {
        if (b.left === 0) { next.push(b); continue; }
        const memory = this.cloneMemory(b.memory || this.memory);
        const logits = this.step(b.toks[b.toks.length - 1], tokens.length + b.toks.length - 1, memory);
        for (const [t, lp] of logSoftmax(logits, COORD0, VOCAB - 1))
          next.push({ lp: b.lp + lp, toks: [...b.toks, t], left: b.left - 1, memory });
      }
      beams = next.sort((a, b) => b.lp - a.lp).slice(0, width);
    }
    return beams.map(b => ({ p: Math.exp(b.lp), tokens: b.toks }));
  }
}
