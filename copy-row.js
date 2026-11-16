const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
function plainCell(value) {
  let text = String(value ?? '').replace(/[\t\r\n]+/g, ' ');
  if (/^[\s\u0000-\u001f]*[=+\-@]/.test(text)) text = "'" + text;
  return text;
}
export function clipboardRow(columns, row) {
  const cells = columns.map(key => plainCell(row[key]));
  return {
    text: cells.map(text => text.includes('"') ? '"' + text.replaceAll('"', '""') + '"' : text).join('\t'),
    html: '<table><tbody><tr>' + cells.map(text => `<td>${escapeHtml(text)}</td>`).join('') + '</tr></tbody></table>',
  };
}
