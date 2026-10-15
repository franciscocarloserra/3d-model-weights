// custom model, front-end only: read just the headers of local .safetensors / .gguf files, nothing is uploaded.
// The record goes to sessionStorage and the page reloads with ?m=__local, so the viewer reads it like any other model.
const GGUF_CHUNK = 4 << 20;  // bytes read per step while walking a gguf header

const compact = files => {  // [[path, header]] -> {name: [dtype, shape, count]}, same as compact() in fetch_tensors.py
  const out = {};
  for (const [f, hdr] of files) {
    const folder = f.includes('/') ? f.slice(0, f.lastIndexOf('/') + 1) : '';
    for (const [name, v] of Object.entries(hdr)) {
      if (name === '__metadata__') continue;
      let seen = false;
      const key = folder + name.split('.').map(p => /^\d+$/.test(p) ? (seen ? '*' : (seen = true, p)) : p).join('.');
      out[key] ??= [v.dtype, v.shape, 0];
      if (out[key][1].join() === v.shape.join()) out[key][2]++;
    }
  }
  return out;
};
const GGML = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K',
  16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS', 19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S', 22: 'IQ2_S', 23: 'IQ4_XS', 24: 'I8', 25: 'I16', 26: 'I32', 27: 'I64', 28: 'F64', 29: 'IQ1_M', 30: 'BF16', 34: 'TQ1_0', 35: 'TQ2_0', 39: 'MXFP4'};
const gguf = async f => {  // walk the binary header in chunks (metadata can hold a multi-MB vocab) -> {name: {dtype, shape}}
  let buf = new Uint8Array(0), dv, pos = 0, base = 0;
  const need = async n => {
    if (pos + n <= buf.length) return;
    const more = new Uint8Array(await f.slice(base + buf.length, base + buf.length + Math.max(n, GGUF_CHUNK)).arrayBuffer());
    if (!more.length) throw new Error('truncated gguf header');
    const b = new Uint8Array(buf.length - pos + more.length); b.set(buf.subarray(pos)); b.set(more, buf.length - pos);
    base += pos; buf = b; pos = 0; dv = new DataView(buf.buffer);
  };
  const u32 = async () => { await need(4); pos += 4; return dv.getUint32(pos - 4, true); };
  const u64 = async () => { await need(8); pos += 8; return Number(dv.getBigUint64(pos - 8, true)); };
  const str = async () => { const n = await u64(); await need(n); pos += n; return new TextDecoder().decode(buf.subarray(pos - n, pos)); };
  const SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8};
  const skip = async type => {
    if (type === 8) { const n = await u64(); await need(n); pos += n; }
    else if (type === 9) { const t = await u32(), n = await u64(); if (SIZE[t]) { await need(SIZE[t] * n); pos += SIZE[t] * n; } else for (let i = 0; i < n; i++) await skip(t); }
    else { await need(SIZE[type]); pos += SIZE[type]; }
  };
  if (await u32() !== 0x46554747) throw new Error(`${f.name}: not a gguf file`);
  await u32();  // version
  const nT = await u64(), nKV = await u64(), out = {};
  for (let i = 0; i < nKV; i++) { await str(); await skip(await u32()); }
  for (let i = 0; i < nT; i++) {
    const name = await str(), nd = await u32(), dims = [];
    for (let j = 0; j < nd; j++) dims.push(await u64());
    const type = await u32(); await u64();  // data offset
    out[name] = {dtype: GGML[type] ?? `T${type}`, shape: dims.reverse()};  // ggml lists the fastest dim first: flip to [out, in]
  }
  return out;
};
const safetensors = async f => {
  const n = Number(new DataView(await f.slice(0, 8).arrayBuffer()).getBigUint64(0, true));
  return JSON.parse(await f.slice(8, 8 + n).text());
};
export const loadLocal = async fs => {
  fs = [...fs].filter(f => /\.(safetensors|gguf)$/.test(f.name));
  if (!fs.length) return;
  const files = await Promise.all(fs.map(async f => [(f.webkitRelativePath || f.name).split('/').slice(1).join('/') || f.name, await (f.name.endsWith('.gguf') ? gguf : safetensors)(f)]));
  const top = fs[0].webkitRelativePath?.split('/')[0];
  const tensors = compact(files);
  sessionStorage.setItem('local', JSON.stringify({id: `local/${top || fs[0].name.replace(/\.(safetensors|gguf)$/, '')}`, tensors, n_files: fs.length, n_tensors: files.reduce((a, [, h]) => a + Object.keys(h).length - ('__metadata__' in h), 0)}));
  location.search = '?m=__local';
};
