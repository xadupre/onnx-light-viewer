const byId = id => document.getElementById(id);
for (const section of document.querySelectorAll('main > section[role="tabpanel"]'))
  section.hidden = section.id !== "prompt-view";
byId("prompt-view").hidden = false;
const promptPanel = createPromptPanel({
  byId,
  postMessage(message) {
    if (message.type === "prompt-info")
      queueMicrotask(() => promptPanel.handleMessage({
        type: "prompt-info", requestId: message.requestId,
        info: { available: true, engines: ["onnxruntime-genai"] },
      }));
    if (message.type === "run-prompt")
      fetch("/result.json").then(response => response.json()).then(result => {
        delete result.profileDirectory;
        result.limitations = [...(result.limitations || []), "Real GenAI runtime output from the repository's tiny causal test model."];
        promptPanel.handleMessage({type:"prompt-result", requestId:message.requestId, result});
      });
  },
});
promptPanel.load();
queueMicrotask(() => {
  byId("prompt-text").value = "H";
  byId("prompt-max-tokens").value = "6";
  byId("prompt-repeat").value = "10";
  byId("prompt-profile-toggle").click();
  if (new URL(location.href).searchParams.has("capture"))
    byId("prompt-form").querySelector('button[type="submit"]').click();
});
