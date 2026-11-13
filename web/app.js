import { loadModel, BOS, EOS, LINE, CIRCLE, ARC, POINT, CONSTRUCTION, COORD0, BINS, MAX_SHAPES, baseType, isConstruction } from './model.js';

// A shape is { type: token, c: [grid values] }, on a 64 x 64 grid with y pointing up (as in the training data).
const board = document.getElementById('board'), ctx = board.getContext('2d');
const SIZE = board.width, CELL = SIZE / BINS;
const px = v => (v + 0.5) * CELL, py = v => (BINS - 1 - v + 0.5) * CELL;
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const NAME = { [LINE]: 'Line', [CIRCLE]: 'Circle', [ARC]: 'Arc', [POINT]: 'Point' };

let model = null, shapes = [], tool = 'line', helper = false, clicks = [], mouse = null;
let guesses = [], chosen = 0, timer = 0;

const toTokens = list => [BOS, ...list.flatMap(s => [s.type, ...s.c.map(v => v + COORD0)])];
const fromTokens = toks => ({ type: toks[0], c: toks.slice(1).map(t => t - COORD0) });

// ---- geometry ----
function arcFromThree(s, e, p) {           // start, end, a point the arc passes through -> shape coordinates
  const circle = circleThrough(s, p, e);
  if (!circle) return null;
  const { x, y, r } = circle, ang = q => Math.atan2(q[1] - y, q[0] - x), TAU = 2 * Math.PI;
  const mod = a => ((a % TAU) + TAU) % TAU;
  let a0 = ang(s), a1 = ang(e);
  if (mod(ang(p) - a0) > mod(a1 - a0)) [a0, a1, s, e] = [a1, a0, e, s];   // keep start -> end anticlockwise
  const mid = a0 + mod(a1 - a0) / 2, snap = v => Math.max(0, Math.min(BINS - 1, Math.round(v)));
  const m = [snap(x + r * Math.cos(mid)), snap(y + r * Math.sin(mid))];
  const same = (a, b) => a[0] === b[0] && a[1] === b[1];
  return same(s, m) || same(m, e) || same(s, e) ? null : [...s, ...m, ...e];
}

function circleThrough(a, b, c) {
  const d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
  if (Math.abs(d) < 1e-6) return null;
  const A = a[0] ** 2 + a[1] ** 2, B = b[0] ** 2 + b[1] ** 2, C = c[0] ** 2 + c[1] ** 2;
  const x = (A * (b[1] - c[1]) + B * (c[1] - a[1]) + C * (a[1] - b[1])) / d;
  const y = (A * (c[0] - b[0]) + B * (a[0] - c[0]) + C * (b[0] - a[0])) / d;
  return { x, y, r: Math.hypot(a[0] - x, a[1] - y) };
}

function tracePath(g, s, mapX, mapY, unit) {
  const b = baseType(s.type), c = s.c;
  g.beginPath();
  if (b === LINE) { g.moveTo(mapX(c[0]), mapY(c[1])); g.lineTo(mapX(c[2]), mapY(c[3])); }
  else if (b === CIRCLE) g.arc(mapX(c[0]), mapY(c[1]), c[2] * unit, 0, 2 * Math.PI);
  else if (b === POINT) g.arc(mapX(c[0]), mapY(c[1]), unit * 0.45, 0, 2 * Math.PI);
  else {
    const circle = circleThrough([c[0], c[1]], [c[2], c[3]], [c[4], c[5]]);
    if (!circle) { g.moveTo(mapX(c[0]), mapY(c[1])); g.lineTo(mapX(c[2]), mapY(c[3])); g.lineTo(mapX(c[4]), mapY(c[5])); return; }
    const { x, y, r } = circle, TAU = 2 * Math.PI, mod = a => ((a % TAU) + TAU) % TAU;
    const a0 = Math.atan2(c[1] - y, c[0] - x), a1 = Math.atan2(c[5] - y, c[4] - x), am = Math.atan2(c[3] - y, c[2] - x);
    const sweep = mod(am - a0) <= mod(a1 - a0) ? mod(a1 - a0) : -mod(a0 - a1);
    for (let i = 0; i <= 40; i++) {
      const a = a0 + sweep * i / 40, X = mapX(x + r * Math.cos(a)), Y = mapY(y + r * Math.sin(a));
      i ? g.lineTo(X, Y) : g.moveTo(X, Y);
    }
  }
}

function paintShape(g, s, color, opts = {}) {
  const { mapX = px, mapY = py, unit = CELL, width = 6, ghost = false } = opts;
  g.save();
  g.strokeStyle = g.fillStyle = color; g.lineWidth = width; g.lineCap = 'round';
  g.setLineDash(ghost ? [width * 3.5, width * 2.5] : isConstruction(s.type) ? [width * 0.5, width * 2.5] : []);
  tracePath(g, s, mapX, mapY, unit);
  baseType(s.type) === POINT ? g.fill() : g.stroke();
  g.restore();
}

// ---- drawing ----
function draft() {                          // the shape being drawn right now, following the mouse
  if (!mouse || !clicks.length) return null;
  const type = { line: LINE, circle: CIRCLE, arc: ARC, point: POINT }[tool] + (helper ? CONSTRUCTION : 0);
  const [a, b] = clicks;
  if (tool === 'line') return a[0] === mouse[0] && a[1] === mouse[1] ? null : { type, c: [...a, ...mouse] };
  if (tool === 'circle') { const r = Math.round(Math.hypot(mouse[0] - a[0], mouse[1] - a[1])); return r ? { type, c: [...a, Math.min(BINS - 1, r)] } : null; }
  if (tool === 'arc') {
    if (!b) return a[0] === mouse[0] && a[1] === mouse[1] ? null : { type: LINE + (helper ? CONSTRUCTION : 0), c: [...a, ...mouse], preview: true };
    const c = arcFromThree(a, b, mouse);
    return c ? { type, c } : null;
  }
  return null;
}

function render() {
  ctx.clearRect(0, 0, SIZE, SIZE);
  ctx.fillStyle = css('--grid');
  for (let i = 0; i < BINS; i += 4) for (let j = 0; j < BINS; j += 4) ctx.fillRect(px(i) - 2, py(j) - 2, 4, 4);
  for (const s of shapes) paintShape(ctx, s, isConstruction(s.type) ? css('--helper') : css('--shape'));
  const d = draft();
  if (d) paintShape(ctx, d, css("--accent"), { width: 5 });
  else if (!clicks.length && guesses[chosen] && guesses[chosen].shape) paintShape(ctx, guesses[chosen].shape, css('--ghost'), { ghost: true });
  if (mouse) { ctx.fillStyle = css('--accent'); ctx.beginPath(); ctx.arc(px(mouse[0]), py(mouse[1]), 5, 0, 7); ctx.fill(); }
}

function renderGuesses() {
  const box = document.getElementById('suggestions');
  box.textContent = '';
  if (!guesses.length) {
    box.innerHTML = `<p class="hint">${shapes.length >= MAX_SHAPES ? 'That is the most shapes the model was trained on (16).' : 'Draw something to get a guess.'}</p>`;
    return;
  }
  guesses.slice(0, 3).forEach((g, i) => {
    const b = document.createElement('button');
    b.className = 'sug' + (i === chosen ? ' on' : '');
    const mini = document.createElement('canvas'); mini.width = mini.height = 88;
    const m = mini.getContext('2d'), u = 88 / BINS, mx = v => (v + 0.5) * u, my = v => (BINS - 1 - v + 0.5) * u;
    for (const s of shapes) paintShape(m, s, css('--helper'), { mapX: mx, mapY: my, unit: u, width: 1.5 });
    if (g.shape) paintShape(m, g.shape, css('--ghost'), { mapX: mx, mapY: my, unit: u, width: 3 });
    const label = g.shape ? `${isConstruction(g.shape.type) ? 'Helper ' : ''}${NAME[baseType(g.shape.type)]}`.replace('Helper L', 'Helper l').replace('Helper C', 'Helper c').replace('Helper A', 'Helper a').replace('Helper P', 'Helper p') : 'Looks finished';
    const text = document.createElement('div'); text.style.flex = '1';
    text.innerHTML = `<div style="display:flex"><b>${i + 1} · ${label}</b><span class="pct">${Math.round(g.p * 100)}%</span></div><div class="bar"><i style="width:${Math.round(g.p * 100)}%"></i></div>`;
    b.append(mini, text);
    b.onmouseenter = () => { chosen = i; render(); for (const [j, el] of [...box.querySelectorAll('.sug')].entries()) el.classList.toggle('on', j === i); };
    b.onclick = () => { chosen = i; accept(); };
    box.append(b);
  });
}

function think() {
  clearTimeout(timer);
  guesses = []; chosen = 0;
  if (!model || !shapes.length || shapes.length >= MAX_SHAPES) { renderGuesses(); render(); return; }
  timer = setTimeout(() => {
    const t0 = performance.now();
    guesses = model.suggest(toTokens(shapes), 5).map(g => ({ p: g.p, shape: g.tokens[0] === EOS ? null : fromTokens(g.tokens) }));
    document.getElementById('status').textContent = `${shapes.length} shape${shapes.length > 1 ? 's' : ''} · guessed in ${Math.round(performance.now() - t0)} ms, in your browser`;
    renderGuesses(); render();
  }, 30);
}

function add(shape) {
  if (shapes.length >= MAX_SHAPES) return;
  shapes.push({ type: shape.type, c: shape.c.slice() });
  think();
}

function accept() {
  const g = guesses[chosen];
  if (g && g.shape && !clicks.length) add(g.shape);
}

function setTool(t) {
  tool = t; clicks = [];
  for (const b of document.querySelectorAll('#tools button')) b.classList.toggle('on', b.dataset.tool === t);
  render();
}

function gridPoint(ev) {
  const r = board.getBoundingClientRect(), clamp = v => Math.max(0, Math.min(BINS - 1, v));
  return [clamp(Math.floor((ev.clientX - r.left) / r.width * BINS)), clamp(BINS - 1 - Math.floor((ev.clientY - r.top) / r.height * BINS))];
}

board.addEventListener('pointermove', ev => { mouse = gridPoint(ev); render(); });
board.addEventListener('pointerleave', () => { mouse = null; render(); });
board.addEventListener('contextmenu', ev => { ev.preventDefault(); clicks = []; render(); });
board.addEventListener('pointerdown', ev => {
  if (ev.button !== 0) return;
  mouse = gridPoint(ev);
  const type = { line: LINE, circle: CIRCLE, arc: ARC, point: POINT }[tool] + (helper ? CONSTRUCTION : 0);
  if (tool === 'point') add({ type, c: mouse });
  else if (!clicks.length) clicks = [mouse];
  else if (tool === 'arc' && clicks.length === 1) { if (clicks[0][0] !== mouse[0] || clicks[0][1] !== mouse[1]) clicks.push(mouse); }
  else {
    const d = draft();
    if (d && !d.preview) { add(d); clicks = tool === 'line' ? [mouse] : []; }   // lines chain from the last end
  }
  render();
});

document.getElementById('tools').addEventListener('click', ev => { const b = ev.target.closest('button'); if (b) setTool(b.dataset.tool); });
const helperBtn = document.getElementById('helper');
const toggleHelper = () => { helper = !helper; helperBtn.classList.toggle('on', helper); render(); };
helperBtn.onclick = toggleHelper;
const undo = () => { if (clicks.length) clicks = []; else shapes.pop(); think(); };
document.getElementById('undo').onclick = undo;
document.getElementById('clear').onclick = () => { shapes = []; clicks = []; think(); };

addEventListener('keydown', ev => {
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  const k = ev.key.toLowerCase();
  if (k === 'tab') { ev.preventDefault(); clicks = []; accept(); }
  else if (k === 'escape') { clicks = []; render(); }
  else if (k === 'l') setTool('line'); else if (k === 'c') setTool('circle');
  else if (k === 'a') setTool('arc'); else if (k === 'p') setTool('point');
  else if (k === 'x') toggleHelper(); else if (k === 'z') undo();
  else if ('123'.includes(k) && guesses[+k - 1]) { chosen = +k - 1; renderGuesses(); render(); }
});

const STARTERS = {
  'Half a box': [{ type: LINE, c: [12, 16, 52, 16] }, { type: LINE, c: [52, 16, 52, 46] }],
  'A circle': [{ type: CIRCLE, c: [32, 32, 18] }],
  'Centre line': [{ type: LINE + CONSTRUCTION, c: [32, 6, 32, 58] }, { type: LINE, c: [32, 12, 50, 12] }],
  'A slot': [{ type: LINE, c: [18, 40, 46, 40] }, { type: ARC, c: [46, 24, 54, 32, 46, 40] }],
};
for (const [name, list] of Object.entries(STARTERS)) {
  const b = document.createElement('button'); b.textContent = name;
  b.onclick = () => { shapes = list.map(s => ({ type: s.type, c: s.c.slice() })); clicks = []; think(); };
  document.getElementById('starters').append(b);
}

setTool('line');
render();
loadModel('.').then(m => {
  model = m;
  document.getElementById('status').textContent = 'Ready. Draw, or pick a start on the right.';
  const start = STARTERS[new URLSearchParams(location.search).get('start')];
  if (start) shapes = start.map(s => ({ type: s.type, c: s.c.slice() }));
  think();
}).catch(err => { document.getElementById('status').textContent = 'Could not load the model: ' + err.message; });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { render(); renderGuesses(); });
