function database() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('benchmark-archive-v1', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('runs', { keyPath: 'id' });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(new Error('请关闭此工具的其他标签页后重试'));
  });
}
async function transaction(mode, operation) {
  const db = await database();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('runs', mode);
    const request = operation(tx.objectStore('runs'));
    tx.oncomplete = () => { db.close(); resolve(request.result); };
    tx.onabort = tx.onerror = () => { db.close(); reject(tx.error || new Error('浏览器存储失败')); };
  });
}
export const saveRecord = record => transaction('readwrite', store => store.put(record));
export const listRecords = () => transaction('readonly', store => store.getAll());
// A run and its attached blobs are a single object, removed in the same transaction.
export const deleteRecord = id => transaction('readwrite', store => store.delete(id));
