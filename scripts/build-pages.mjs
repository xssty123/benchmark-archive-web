import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export async function buildPages({ source, output, version }) {
  if (!/^[a-f0-9]{40}$/.test(version ?? '')) throw new Error('A full commit SHA is required');
  const assets = path.join(output, 'assets', version);
  await fs.mkdir(assets, { recursive: true });
  await fs.cp(source, assets, { recursive: true });
  // Preserve old entry/download URLs while new clients load only versioned assets.
  await fs.cp(source, output, { recursive: true });
  let html = await fs.readFile(path.join(source, 'index.html'), 'utf8');
  html = html.replace('name="app-version" content="development"', `name="app-version" content="${version}"`);
  html = html.replace('href="./style.css"', `href="./assets/${version}/style.css"`);
  await fs.writeFile(path.join(output, 'index.html'), html);
  await fs.writeFile(path.join(output, 'version.json'), JSON.stringify({ version }) + '\n');
  await fs.writeFile(path.join(output, 'app.js'), "import './boot.js';\n");
  await fs.writeFile(path.join(output, '.nojekyll'), '');
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const root = fileURLToPath(new URL('../', import.meta.url));
  await buildPages({ source: path.join(root, 'public'), output: path.join(root, '_site'), version: process.env.GITHUB_SHA });
}
