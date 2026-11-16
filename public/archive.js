const encoder = new TextEncoder();
const crcTable = Array.from({ length: 256 }, (_, n) => {
  for (let i = 0; i < 8; i++) n = (n & 1) ? 0xedb88320 ^ (n >>> 1) : n >>> 1;
  return n >>> 0;
});
function crc32(bytes) {
  let n = 0xffffffff;
  for (const b of bytes) n = crcTable[(n ^ b) & 255] ^ (n >>> 8);
  return (n ^ 0xffffffff) >>> 0;
}
// ZIP store mode keeps original file bytes; no external dependency or network request.
export function buildZip(files) {
  const local = [], central = [];
  let offset = 0, centralSize = 0;
  for (const file of files) {
    if (!file.name || /(^\/|\\|(^|\/)\.\.(\/|$))/.test(file.name)) throw new Error('非法归档路径');
    const name = encoder.encode(file.name), bytes = file.bytes;
    const crc = crc32(bytes);
    const header = new Uint8Array(30 + name.length), h = new DataView(header.buffer);
    h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint16(6, 0x800, true);
    h.setUint16(12, 33, true); // 1980-01-01; original timestamps are not inferred
    h.setUint32(14, crc, true); h.setUint32(18, bytes.length, true); h.setUint32(22, bytes.length, true);
    h.setUint16(26, name.length, true); header.set(name, 30);
    local.push(header, bytes);
    const directory = new Uint8Array(46 + name.length), d = new DataView(directory.buffer);
    d.setUint32(0, 0x02014b50, true); d.setUint16(4, 20, true); d.setUint16(6, 20, true);
    d.setUint16(8, 0x800, true); d.setUint16(14, 33, true); d.setUint32(16, crc, true);
    d.setUint32(20, bytes.length, true); d.setUint32(24, bytes.length, true);
    d.setUint16(28, name.length, true); d.setUint32(42, offset, true); directory.set(name, 46);
    central.push(directory); centralSize += directory.length; offset += header.length + bytes.length;
  }
  const end = new Uint8Array(22), e = new DataView(end.buffer);
  e.setUint32(0, 0x06054b50, true); e.setUint16(8, files.length, true); e.setUint16(10, files.length, true);
  e.setUint32(12, centralSize, true); e.setUint32(16, offset, true);
  return new Blob([...local, ...central, end], { type: 'application/zip' });
}

function csvCell(value) {
  let text = String(value ?? '');
  // Excel formula injection: preserve data as literal text, including leading controls.
  if (/^[\s\u0000-\u001f]*[=+\-@]/.test(text)) text = "'" + text;
  return '"' + text.replaceAll('"', '""') + '"';
}
export function makeCsv(columns, row) {
  return '\ufeff' + [columns.map(csvCell).join(','), columns.map(key => csvCell(row[key])).join(',')].join('\r\n') + '\r\n';
}
export async function zipRecord(record) {
  const prefix = record.id + '/';
  const files = [
    { name: prefix + 'run.log', bytes: new Uint8Array(await record.log.arrayBuffer()) },
  ];
  const links = Object.fromEntries(['环境信息', '日志存档', 'profiling采集'].filter(key => record.row?.[key]).map(key => [key, record.row[key]]));
  if (Object.keys(links).length) files.push({ name: prefix + 'material-links.json', bytes: encoder.encode(JSON.stringify(links, null, 2) + '\n') });
  // Preserve attachments on older records; new records only store the log and links.
  if (record.environment) files.push({ name: prefix + 'environment.txt', bytes: new Uint8Array(await record.environment.arrayBuffer()) });
  for (const [index, file] of (record.profiles || []).entries()) {
    const safe = file.name.replace(/[\\/\x00-\x1f]/g, '_');
    files.push({ name: prefix + `profiling/${index + 1}-${safe}`, bytes: new Uint8Array(await file.arrayBuffer()) });
  }
  return buildZip(files);
}
