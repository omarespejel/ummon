// Glue for ummon_web_verifier.wasm, shared by the page and node_check.mjs. No imports, no network.
//
// verify(statementText, files) copies the statement and the files ([{name, bytes: Uint8Array}]) into the module's
// memory, calls ummon_verify, and returns {accepted, report, ms}; ms times the ummon_verify call alone.

function packFiles(files) {
  const enc = new TextEncoder();
  const parts = files.map((f) => [enc.encode(f.name), f.bytes]);
  const total = parts.reduce((n, [name, bytes]) => n + 8 + name.length + bytes.length, 0);
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  let at = 0;
  for (const field of parts.flat()) {
    view.setUint32(at, field.length, true);
    out.set(field, at + 4);
    at += 4 + field.length;
  }
  return out;
}

export async function loadVerifier(wasmBytes) {
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const x = instance.exports;

  function put(bytes) {
    const ptr = x.ummon_alloc(bytes.length);
    new Uint8Array(x.memory.buffer, ptr, bytes.length).set(bytes);
    return ptr;
  }

  function verify(statementText, files) {
    const statement = new TextEncoder().encode(statementText);
    const packed = packFiles(files);
    const sp = put(statement);
    const fp = put(packed);
    const t0 = performance.now();
    const ok = x.ummon_verify(sp, statement.length, fp, packed.length);
    const ms = performance.now() - t0;
    const report = new TextDecoder().decode(
      new Uint8Array(x.memory.buffer, x.ummon_result_ptr(), x.ummon_result_len()),
    );
    x.ummon_free(sp, statement.length);
    x.ummon_free(fp, packed.length);
    return { accepted: ok === 1, report, ms };
  }

  return { verify, memoryBytes: () => x.memory.buffer.byteLength };
}
