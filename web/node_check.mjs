// Run the wasm verifier under Node on a bundle directory: print the report, then, for an accepted bundle, 3 warm-up
// runs and `runs` timed runs (min and median, ms), the module's memory size and the process peak RSS.
//
// Usage: node node_check.mjs <ummon_web_verifier.wasm> <bundle_dir> <statement.txt> [runs]
// (default 30 runs; 0 prints the report only). A trap is reported as a reject with its message.

import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { loadVerifier } from './ummon.js';

const [wasmPath, dir, statementPath, runsArg] = process.argv.slice(2);
if (!statementPath) {
  console.error('usage: node node_check.mjs <wasm> <bundle_dir> <statement.txt> [runs]');
  process.exit(2);
}
const runs = runsArg === undefined ? 30 : Number(runsArg);
const files = readdirSync(dir, { withFileTypes: true })
  .filter((e) => e.isFile())
  .map((e) => e.name)
  .sort()
  .map((name) => ({ name, bytes: new Uint8Array(readFileSync(join(dir, name))) }));
const statement = readFileSync(statementPath, 'utf8');
const verifier = await loadVerifier(readFileSync(wasmPath));

let first;
try {
  first = verifier.verify(statement, files);
} catch (e) {
  console.log(`result reject\nreason wasm trap: ${e.message}`);
  process.exit(0);
}
process.stdout.write(first.report);
if (!first.accepted || runs === 0) process.exit(0);

for (let i = 0; i < 3; i++) verifier.verify(statement, files);
const ms = [];
for (let i = 0; i < runs; i++) {
  const r = verifier.verify(statement, files);
  if (!r.accepted) throw new Error(`run ${i} rejected: ${r.report}`);
  ms.push(r.ms);
}
ms.sort((a, b) => a - b);
console.log(
  `wasm_verify_ms min ${ms[0].toFixed(3)} median ${ms[Math.floor(runs / 2)].toFixed(3)} runs ${runs} warmups 3 ` +
    `first ${first.ms.toFixed(3)}`,
);
console.log(
  `wasm_memory_bytes ${verifier.memoryBytes()} node_peak_rss_kib ${process.resourceUsage().maxRSS}`,
);
