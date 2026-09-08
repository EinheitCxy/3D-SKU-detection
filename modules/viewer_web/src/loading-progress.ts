/** Report response-body progress so a stalled transfer is identifiable. */
export function progressFetch(fetcher: typeof fetch, report: (message: string) => void): typeof fetch {
  const pending = new Map<number, string>();
  let requested = 0;
  let completed = 0;
  let bytes = 0;
  const update = () => report(
    `正在加载 ${completed}/${requested} 个文件 · 已接收 ${(bytes / 1_000_000).toFixed(1)} MB`
    + (pending.size ? ` · 等待：${[...pending.values()].join(', ')}` : ' · 正在解析数据'),
  );
  return async (input, init) => {
    const id = ++requested;
    const url = input instanceof Request ? input.url : String(input);
    pending.set(id, url.split('/').pop() || url);
    update();
    try {
      const response = await fetcher(input, init);
      if (!response.ok || response.body === null) {
        pending.delete(id);
        update();
        return response;
      }
      const body = response.body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
        transform(chunk, controller) {
          bytes += chunk.byteLength;
          update();
          controller.enqueue(chunk);
        },
        flush() {
          completed++;
          pending.delete(id);
          update();
        },
      }));
      return new Response(body, { status: response.status, statusText: response.statusText, headers: response.headers });
    } catch (error) {
      pending.delete(id);
      update();
      throw error;
    }
  };
}
