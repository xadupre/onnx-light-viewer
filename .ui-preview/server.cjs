const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const root = path.resolve(__dirname, "..");
const { webviewHtml } = require("../dist/webview");
const html = webviewHtml(name => `/media/${name}`, "http://127.0.0.1:8765", "nonce", "preview")
  .replace(/<meta[^>]+Content-Security-Policy[^>]+>/, "")
  .replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, "")
  .replace("</head>", `<style>
:root {
  --vscode-foreground:#d4d4d4; --vscode-editor-background:#1e1e1e;
  --vscode-sideBar-background:#252526; --vscode-panel-border:#454545;
  --vscode-input-background:#303031; --vscode-input-foreground:#ddd;
  --vscode-input-border:#666; --vscode-focusBorder:#007fd4;
  --vscode-button-background:#0e639c; --vscode-button-foreground:#fff;
  --vscode-descriptionForeground:#aaa; --vscode-charts-blue:#3794ff;
}
body { background:#1e1e1e; color:#d4d4d4; font-family:Arial,sans-serif; font-size:14px; }
main > nav { display:none !important; }
#prompt-view { max-width:1100px; margin:auto; }
.profiling-plots svg { max-width:650px; }
</style></head>`)
  .replace("</body>", `<script src="/media/profiling-panel.js"></script>
<script src="/media/prompt-panel.js"></script><script src="/preview.js"></script></body>`);
http.createServer((request, response) => {
  const url = new URL(request.url, "http://localhost");
  if (url.pathname === "/") {
    response.setHeader("Content-Type", "text/html"); response.end(html); return;
  }
  if (url.pathname === "/result.json") {
    response.setHeader("Content-Type", "application/json");
    response.end(fs.readFileSync(path.join(__dirname, "result.json"))); return;
  }
  const file = url.pathname === "/preview.js" ? path.join(__dirname, "preview.js")
    : path.join(root, url.pathname);
  if (!file.startsWith(root + path.sep) || !fs.existsSync(file)) {
    response.writeHead(404); response.end(); return;
  }
  response.setHeader("Content-Type", file.endsWith(".css") ? "text/css" : "text/javascript");
  response.end(fs.readFileSync(file));
}).listen(8765, "0.0.0.0");
