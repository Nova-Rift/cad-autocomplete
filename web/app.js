import { loadModel, BOS, EOS, LINE, CIRCLE, ARC, POINT, CONSTRUCTION, COORD0, BINS, MAX_SHAPES, baseType, isConstruction } from './model.js';

// A shape is { type: token, c: [grid values] }, on a 64 x 64 grid with y pointing up (as in the training data).
const board = document.getElementById('board'), ctx = board.getContext('2d');
const SIZE = board.width, CELL = SIZE / BINS;
const px = v => (v + 0.5) * CELL, py = v => (BINS - 1 - v + 0.5) * CELL;
const css = name => getComputedStyle(board).getPropertyValue(name.replace('--', '--sk-')).trim();
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
  // every guess sits on the board: the chosen one strong, the others faint, each with its number
  const shown = guesses.slice(0, 3).map((g, i) => [g, i]).filter(([g]) => g.shape);
  for (const [g, i] of shown) if (i !== chosen) { ctx.globalAlpha = 0.38; paintShape(ctx, g.shape, css('--ghost'), { ghost: true, width: 4 }); ctx.globalAlpha = 1; }
  for (const [g, i] of shown) if (i === chosen) paintShape(ctx, g.shape, css('--ghost'), { ghost: true, width: 7 });
  const d = draft();
  if (d) paintShape(ctx, d, css("--accent"), { width: 5 });
  for (const [g, i] of shown) badge(g.shape, i);
  if (mouse) { ctx.fillStyle = css('--accent'); ctx.beginPath(); ctx.arc(px(mouse[0]), py(mouse[1]), 5, 0, 7); ctx.fill(); }
}

function anchor(s) {                         // where a guess's number sits
  const b = baseType(s.type), c = s.c;
  if (b === LINE) return [(c[0] + c[2]) / 2, (c[1] + c[3]) / 2];
  if (b === CIRCLE) return [c[0], c[1] + c[2]];
  if (b === ARC) return [c[2], c[3]];
  return [c[0], c[1] + 2];
}

function badge(s, i) {
  const [x, y] = anchor(s), X = px(x), Y = py(y), on = i === chosen, r = on ? 19 : 15;
  ctx.save();
  ctx.globalAlpha = on ? 1 : 0.75;
  ctx.fillStyle = on ? css('--ghost') : css('--panel'); ctx.strokeStyle = css('--ghost'); ctx.lineWidth = 3;
  ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fill(); ctx.stroke();
  ctx.fillStyle = on ? css('--panel') : css('--ghost'); ctx.font = `700 ${on ? 22 : 18}px system-ui, sans-serif`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(String(i + 1), X, Y + 1);
  ctx.restore();
}

const labelOf = g => g.shape ? (isConstruction(g.shape.type) ? 'Helper ' + NAME[baseType(g.shape.type)].toLowerCase() : NAME[baseType(g.shape.type)]) : 'Looks finished';

function renderPick() {                      // the strip on the board: step through the guesses, take one
  const pick = document.getElementById('pick'), n = Math.min(3, guesses.length);
  if (!pick) return;
  pick.hidden = !n;
  if (!n) return;
  const g = guesses[chosen];
  document.getElementById('pick-label').textContent = `Guess ${chosen + 1} of ${n} · ${labelOf(g)} · ${Math.round(g.p * 100)}%`;
  document.getElementById('take').disabled = !g.shape;
}

function cycle(step) {
  const n = Math.min(3, guesses.length);
  if (!n) return;
  chosen = (chosen + step + n) % n;
  renderGuesses(); render();
}

function renderGuesses() {
  renderPick();
  const box = document.getElementById('suggestions');
  box.textContent = '';
  if (!guesses.length) {
    box.innerHTML = `<p class="sk-hint">${shapes.length >= MAX_SHAPES ? 'That is the most shapes the model was trained on (16).' : 'Draw something to get a guess.'}</p>`;
    return;
  }
  guesses.slice(0, 3).forEach((g, i) => {
    const b = document.createElement('button');
    b.className = 'sk-sug' + (i === chosen ? ' on' : '');
    const mini = document.createElement('canvas'); mini.width = mini.height = 88;
    const m = mini.getContext('2d'), u = 88 / BINS, mx = v => (v + 0.5) * u, my = v => (BINS - 1 - v + 0.5) * u;
    for (const s of shapes) paintShape(m, s, css('--helper'), { mapX: mx, mapY: my, unit: u, width: 1.5 });
    if (g.shape) paintShape(m, g.shape, css('--ghost'), { mapX: mx, mapY: my, unit: u, width: 3 });
    const label = labelOf(g);
    const text = document.createElement('div'); text.style.flex = '1';
    text.innerHTML = `<div style="display:flex"><b>${i + 1} · ${label}</b><span class="sk-pct">${Math.round(g.p * 100)}%</span></div><div class="sk-bar"><i style="width:${Math.round(g.p * 100)}%"></i></div>`;
    b.append(mini, text);
    b.onmouseenter = () => { chosen = i; render(); renderPick(); for (const [j, el] of [...box.querySelectorAll('.sk-sug')].entries()) el.classList.toggle('on', j === i); };
    b.onclick = () => { chosen = i; accept(); };
    box.append(b);
  });
}

// What the person has already fixed by clicking, as {position in the new shape: token}: the kind of shape and
// where it starts (for an arc, also where it ends). The guesses must agree with it.
function fixedSoFar() {
  if (!clicks.length) return {};
  const type = { line: LINE, circle: CIRCLE, arc: ARC, point: POINT }[tool] + (helper ? CONSTRUCTION : 0);
  const f = { 0: type, 1: clicks[0][0] + COORD0, 2: clicks[0][1] + COORD0 };
  if (tool === 'arc' && clicks[1]) { f[5] = clicks[1][0] + COORD0; f[6] = clicks[1][1] + COORD0; }
  return f;
}

function think() {
  clearTimeout(timer);
  guesses = []; chosen = 0;
  if (!model || (!shapes.length && !clicks.length) || shapes.length >= MAX_SHAPES) { renderGuesses(); render(); return; }
  timer = setTimeout(() => {
    const t0 = performance.now(), fixed = fixedSoFar();
    guesses = model.suggest(toTokens(shapes), 5, fixed).map(g => ({ p: g.p, shape: g.tokens[0] === EOS ? null : fromTokens(g.tokens) }));
    document.getElementById('status').textContent = `${shapes.length} shape${shapes.length === 1 ? '' : 's'} · ${clicks.length ? 'guesses follow what you started' : 'guessed'} in ${Math.round(performance.now() - t0)} ms, in your browser`;
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
  if (!g || !g.shape) return;
  // taking a guess part-way through a line carries on from that line's far end, as drawing does
  const s = g.shape, from = clicks[0], chain = from && tool === 'line' && baseType(s.type) === LINE;
  clicks = chain ? [s.c[0] === from[0] && s.c[1] === from[1] ? [s.c[2], s.c[3]] : [s.c[0], s.c[1]]] : [];
  add(s);
}

function setTool(t) {
  const had = clicks.length;
  tool = t; clicks = [];
  for (const b of document.querySelectorAll('#tools button')) b.classList.toggle('on', b.dataset.tool === t);
  had ? think() : render();
}

function gridPoint(ev) {
  const r = board.getBoundingClientRect(), clamp = v => Math.max(0, Math.min(BINS - 1, v));
  return [clamp(Math.floor((ev.clientX - r.left) / r.width * BINS)), clamp(BINS - 1 - Math.floor((ev.clientY - r.top) / r.height * BINS))];
}

board.addEventListener('pointermove', ev => { mouse = gridPoint(ev); render(); });
board.addEventListener('pointerleave', () => { mouse = null; render(); });
board.addEventListener('contextmenu', ev => { ev.preventDefault(); if (clicks.length) { clicks = []; think(); } });
board.addEventListener('pointerdown', ev => {
  if (ev.button !== 0) return;
  mouse = gridPoint(ev);
  const type = { line: LINE, circle: CIRCLE, arc: ARC, point: POINT }[tool] + (helper ? CONSTRUCTION : 0);
  if (tool === 'point') add({ type, c: mouse });
  else if (!clicks.length) { clicks = [mouse]; think(); }                       // guesses now follow this start
  else if (tool === 'arc' && clicks.length === 1) { if (clicks[0][0] !== mouse[0] || clicks[0][1] !== mouse[1]) { clicks.push(mouse); think(); } }
  else {
    const d = draft();
    if (d && !d.preview) { clicks = tool === 'line' ? [mouse] : []; add(d); }   // lines chain from the last end
  }
  render();
});

document.getElementById('tools').addEventListener('click', ev => { const b = ev.target.closest('button'); if (b) setTool(b.dataset.tool); });
const helperBtn = document.getElementById('helper');
const toggleHelper = () => { helper = !helper; helperBtn.classList.toggle('on', helper); clicks.length ? think() : render(); };
helperBtn.onclick = toggleHelper;
const undo = () => { if (clicks.length) clicks = []; else shapes.pop(); think(); };
document.getElementById('undo').onclick = undo;
document.getElementById('clear').onclick = () => { shapes = []; clicks = []; think(); };

addEventListener('keydown', ev => {
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  const at = document.activeElement, typing = at && /^(input|textarea|select)$/i.test(at.tagName);
  const ours = !at || at === document.body || !!at.closest('.sk');      // leave Tab alone in the site's own menus
  if (typing) return;
  const k = ev.key.toLowerCase();
  if (k === 'tab') { if (!ours || !guesses[chosen] || !guesses[chosen].shape) return; ev.preventDefault(); accept(); }
  else if (k === 'escape') { if (clicks.length) { clicks = []; think(); } }
  else if (k === 'arrowright' || k === 'arrowdown') { if (guesses.length) { ev.preventDefault(); cycle(1); } }
  else if (k === 'arrowleft' || k === 'arrowup') { if (guesses.length) { ev.preventDefault(); cycle(-1); } }
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

document.getElementById('prev').onclick = () => cycle(-1);
document.getElementById('next').onclick = () => cycle(1);
document.getElementById('take').onclick = accept;
setTool('line');
render();
loadModel(new URL('.', import.meta.url).href.replace(/\/$/, '')).then(m => {
  model = m;
  document.getElementById('status').textContent = 'Ready. Draw, or pick a start on the right.';
  const v = document.getElementById('version');
  if (v) v.textContent = `This copy of the model had ${m.trainedSteps.toLocaleString()} training steps.`;
  const start = STARTERS[new URLSearchParams(location.search).get('start')];
  if (start) shapes = start.map(s => ({ type: s.type, c: s.c.slice() }));
  const q = new URLSearchParams(location.search), at = (q.get('click') || '').split(',').map(Number);   // for checking the page: ?tool=circle&click=20,40
  if (q.get('tool')) setTool(q.get('tool'));
  if (at.length === 2 && at.every(Number.isFinite)) clicks = [at];
  think();
}).catch(err => { document.getElementById('status').textContent = 'Could not load the model: ' + err.message; });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { render(); renderGuesses(); });
new MutationObserver(() => { render(); renderGuesses(); }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
