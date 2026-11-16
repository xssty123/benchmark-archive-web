import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { buildPages } from '../scripts/build-pages.mjs';

const source = new URL('../public/', import.meta.url);
const version = 'a'.repeat(40);
const loader = await fs.readFile(new URL('boot.js', source), 'utf8');
// Run the shipped loader with browser I/O replaced; does not simulate rendering.
async function boot({ pageVersion = version, manifestVersion = version, href = 'https://example.test/project/', failFetch = false, failImport = false } = {}) {
  const elements = new Map();
  const document = { querySelector(selector) {
    if (!elements.has(selector)) elements.set(selector, { textContent: '', content: pageVersion, classList: { remove() {} } });
    return elements.get(selector);
  } };
  const result = { elements, imported: [] };
  const location = { href, replace(url) { result.redirect = url; } };
  const fetch = async (url, options) => {
    result.request = { url, options };
    return { ok: !failFetch, json: async () => ({ version: manifestVersion }) };
  };
  const importModule = async url => {
    result.imported.push(url);
    assert.equal(elements.get('#build-version').textContent, ''); // only label a loaded version
    if (failImport) throw new Error('module unavailable');
  };
  const code = loader.replaceAll('import.meta.url', JSON.stringify('https://example.test/project/boot.js')).replace('await import(', 'await importModule(');
  await new Function('document', 'location', 'fetch', 'importModule', `return (async () => {${code}})()`)(document, location, fetch, importModule);
  return result;
}

test('deployment pins the full parser graph and schema to one commit', async t => {
  const output = await fs.mkdtemp(path.join(os.tmpdir(), 'benchmark-pages-'));
  t.after(() => fs.rm(output, { recursive: true, force: true }));
  await buildPages({ source: fileURLToPath(source), output, version });
  const assets = path.join(output, 'assets', version);
  const html = await fs.readFile(path.join(output, 'index.html'), 'utf8');
  assert.ok(html.includes(`content="${version}"`));
  assert.ok(html.includes(`./assets/${version}/style.css`));
  assert.ok(html.includes('src="./boot.js"'));
  assert.deepEqual(JSON.parse(await fs.readFile(path.join(output, 'version.json'))), { version });
  await fs.writeFile(path.join(output, 'package.json'), '{"type":"module"}');
  const parser = await import(pathToFileURL(path.join(assets, 'parser.js')));
  const parsed = parser.parseSFT('VeOmni\n' + JSON.stringify({ model: { model_path: '/models/MiniMax-H3/FL2VA/transformer' }, data: { data_type: 'diffusion' }, train: {} }));
  assert.equal(parsed.configuration.fields['模型'], 'MiniMax-H3');
  for (const file of (await fs.readdir(source)).filter(file => file.endsWith('.js'))) {
    const built = await fs.readFile(path.join(assets, file), 'utf8');
    assert.equal(built, await fs.readFile(new URL(file, source), 'utf8'));
    for (const match of built.matchAll(/from ['"](.+?)['"]/g)) {
      assert.ok(match[1].startsWith('./'));
      await fs.access(path.join(assets, match[1]));
    }
  }
  assert.match(await fs.readFile(path.join(assets, 'app.js'), 'utf8'), /new URL\('\.\/columns.json', import.meta.url\)/);
  await fs.access(path.join(assets, 'columns.json'));
  await fs.access(path.join(assets, 'scripts/collect_environment.py'));
});

test('loader bypasses manifest cache and loads versioned app, CSS and download', async () => {
  const r = await boot();
  assert.equal(r.request.options.cache, 'no-store');
  assert.ok(r.request.url.searchParams.get('t'));
  assert.deepEqual(r.imported, [`https://example.test/project/assets/${version}/app.js`]);
  assert.equal(r.elements.get('#build-version').textContent, '版本 aaaaaaa');
  assert.equal(r.elements.get('#build-version').title, version);
  assert.match(r.elements.get('.environment-download').href, /assets\/a{40}\/scripts\/collect_environment.py$/);
});

test('cached HTML redirects to latest release before initializing any app', async () => {
  const r = await boot({ pageVersion: 'b'.repeat(40) });
  assert.equal(new URL(r.redirect).searchParams.get('release'), version);
  assert.deepEqual(r.imported, []);
  const repeated = await boot({ pageVersion: 'b'.repeat(40), href: `https://example.test/project/?release=${version}` });
  assert.equal(repeated.redirect, undefined);
  assert.match(repeated.elements.get('#alert').textContent, /强制刷新/);
});

test('failed or invalid releases show errors without falling back to old app', async () => {
  for (const options of [{ failFetch: true }, { manifestVersion: '../old' }, { failImport: true }]) {
    const r = await boot(options);
    assert.equal(r.elements.get('#build-version').textContent, '加载失败');
    assert.match(r.elements.get('#alert').textContent, /页面加载失败/);
  }
});

test('development loads local sources without a deployment build', async () => {
  const r = await boot({ pageVersion: 'development', manifestVersion: 'development' });
  assert.deepEqual(r.imported, ['https://example.test/project/app.js']);
  assert.equal(r.elements.get('#build-version').textContent, '开发版');
});
