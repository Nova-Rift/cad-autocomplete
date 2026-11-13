// Proves the JavaScript model gives the same answers as the PyTorch model.
//   node test/parity.mjs
import { readFileSync } from 'node:fs';
import { buildModel, VOCAB } from '../web/model.js';

const dir = new URL('../web/', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('weights.json', dir)));
const bin = readFileSync(new URL('weights.bin', dir));
const model = buildModel(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const cases = JSON.parse(readFileSync(new URL('parity.json', dir)));

let worst = 0, sameTop = 0, sameAll = 0, ms = 0;
for (const c of cases) {
  model.reset();
  const logits = model.read(c.prefix);
  for (let i = 0; i < VOCAB; i++) worst = Math.max(worst, Math.abs(logits[i] - c.logits[i]));
  const t0 = performance.now();
  const got = model.suggest(c.prefix, 5);
  ms += performance.now() - t0;
  if (got[0].tokens.join() === c.suggestions[0].tokens.join()) sameTop++;
  if (got.map(g => g.tokens.join()).join('|') === c.suggestions.map(g => g.tokens.join()).join('|')) sameAll++;
}
console.log(`cases ${cases.length}; largest score difference ${worst.toFixed(4)}; same top guess ${sameTop}/${cases.length}; ` +
  `same five guesses ${sameAll}/${cases.length}; ${(ms / cases.length).toFixed(0)} ms per suggestion`);
if (worst > 0.05 || sameTop < cases.length) { console.error('FAIL'); process.exit(1); }
console.log('PASS');
