// Keep this loader dependency-free: every application module is versioned together.
const base = new URL('./', import.meta.url);
const status = document.querySelector('#build-version');
try {
  const manifest = new URL('version.json', base);
  manifest.searchParams.set('t', Date.now());
  const response = await fetch(manifest, { cache: 'no-store' });
  if (!response.ok) throw new Error('版本信息无法读取');
  const { version } = await response.json();
  if (version !== 'development' && !/^[a-f0-9]{40}$/.test(version)) {
    throw new Error('版本信息无效');
  }
  const pageVersion = document.querySelector('meta[name="app-version"]')?.content;
  if (version !== 'development' && pageVersion !== version) {
    // A cached HTML document must be refreshed too, before using new UI code.
    const fresh = new URL(location.href);
    if (fresh.searchParams.get('release') === version) throw new Error('页面版本未更新，请强制刷新后重试');
    fresh.searchParams.set('release', version);
    fresh.searchParams.set('t', Date.now());
    location.replace(fresh.href);
  } else {
    const assets = version === 'development' ? base : new URL(`assets/${version}/`, base);
    document.querySelector('link[rel="stylesheet"]').href = new URL('style.css', assets).href;
    document.querySelector('.environment-download').href = new URL('scripts/collect_environment.py', assets).href;
    await import(new URL('app.js', assets).href);
    if (status) {
      status.textContent = version === 'development' ? '开发版' : `版本 ${version.slice(0, 7)}`;
      status.title = version;
    }
  }
} catch (error) {
  if (status) status.textContent = '加载失败';
  const alert = document.querySelector('#alert');
  alert.textContent = `页面加载失败：${error.message}。请刷新后重试`;
  alert.classList.remove('hidden');
}
