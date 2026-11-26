// The page's script: load the verifier, verify the chosen files against the statement, show the result.
import { loadVerifier } from './ummon.js';

const $ = (id) => document.getElementById(id);
let verifier;
try {
  const wasm = await (await fetch('ummon_web_verifier.wasm')).arrayBuffer();
  verifier = await loadVerifier(wasm);
  $('status').textContent = 'Ready.';
  $('run').disabled = false;
} catch (e) {
  $('status').textContent = `Could not load the verifier: ${e.message}`;
}

$('run').addEventListener('click', async () => {
  const picked = [...$('files').files];
  if (picked.length === 0) {
    $('status').textContent = 'Choose the bundle files first.';
    return;
  }
  $('run').disabled = true;
  $('status').textContent = 'Reading files…';
  const files = await Promise.all(picked.map(async (f) => ({ name: f.name, bytes: new Uint8Array(await f.arrayBuffer()) })));
  $('status').textContent = 'Verifying…';
  await new Promise((r) => setTimeout(r, 20));
  let result;
  try {
    result = verifier.verify($('statement').value, files);
  } catch (e) {
    result = { accepted: false, report: `result reject\nreason wasm trap: ${e.message}\n(reload the page before the next check)`, ms: NaN };
  }
  const bytes = files.reduce((n, f) => n + f.bytes.length, 0);
  $('verdict').textContent = result.accepted ? 'Accepted' : 'Rejected';
  $('verdict').className = result.accepted ? 'accept' : 'reject';
  $('report').textContent = `${result.report}verify_ms ${result.ms.toFixed(1)}\nfiles ${files.length} (${bytes} bytes)\n` +
    `wasm_memory_bytes ${verifier.memoryBytes()}\n`;
  $('report').hidden = false;
  $('status').textContent = `Done in ${result.ms.toFixed(1)} ms.`;
  $('run').disabled = false;
});
