import { createReadStream, existsSync } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, join, normalize, resolve } from "node:path";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const preferredRoot = resolve(projectRoot, "dist");
const staticRoot = existsSync(preferredRoot) ? preferredRoot : resolve(projectRoot, "src");
const port = Number(process.env.PORT ?? 4173);
const host = process.env.HOST ?? "127.0.0.1";

const mimeTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
};

const server = createServer(async (request, response) => {
  try {
    const requestUrl = new URL(request.url ?? "/", `http://${host}:${port}`);
    const safePath = normalize(decodeURIComponent(requestUrl.pathname)).replace(/^(\.\.[/\\])+/, "");
    const requestedFile = safePath === "/" ? "index.html" : safePath.replace(/^[/\\]/, "");
    let filePath = resolve(staticRoot, requestedFile);

    if (!filePath.startsWith(staticRoot)) {
      response.writeHead(403);
      response.end("Доступ запрещён");
      return;
    }

    const fileStat = await stat(filePath).catch(() => null);
    if (!fileStat?.isFile()) {
      filePath = join(staticRoot, "index.html");
    }

    response.writeHead(200, {
      "Content-Type": mimeTypes[extname(filePath)] ?? "application/octet-stream",
      "Cache-Control": "no-store",
    });
    createReadStream(filePath).pipe(response);
  } catch (error) {
    response.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
    response.end(error instanceof Error ? error.message : "Ошибка сервера");
  }
});

server.listen(port, host, () => {
  console.log(`Reqcore ATS UI запущен: http://${host}:${port}`);
  console.log(`Статические файлы: ${staticRoot}`);
});
