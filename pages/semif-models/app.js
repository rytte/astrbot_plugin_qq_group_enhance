const bridge = window.AstrBotPluginPage;
const modelsElement = document.getElementById("models");
const noticeElement = document.getElementById("notice");

function showNotice(message, error = false) {
  noticeElement.hidden = !message;
  noticeElement.textContent = message || "";
  noticeElement.className = error ? "notice error" : "notice";
}

function addText(parent, tag, value, className = "") {
  const element = document.createElement(tag);
  element.textContent = value;
  if (className) element.className = className;
  parent.append(element);
  return element;
}

function addSelect(card, key, selected) {
  const button = addText(card, "button", selected ? "当前模型" : "选择模型");
  button.disabled = selected;
  button.addEventListener("click", () => selectModel(key));
}

function render(data) {
  modelsElement.replaceChildren();
  const custom = document.createElement("article");
  custom.className = `model-card ${data.selected === "custom" ? "selected" : ""}`;
  addText(custom, "h2", "自定义模型");
  addText(custom, "p", "使用插件配置中的 SemIf 模型目录或 Hugging Face ID，以及固定模型版本。");
  const customActions = addText(custom, "div", "", "actions");
  addSelect(customActions, "custom", data.selected === "custom");
  modelsElement.append(custom);

  for (const model of data.models || []) {
    const card = document.createElement("article");
    card.className = `model-card ${model.selected ? "selected" : ""}`;
    const heading = addText(card, "div", "", "heading");
    const name = addText(heading, "div", "");
    addText(name, "h2", model.label);
    addText(name, "code", model.repo_id);
    const stateText = {
      ready: "已下载",
      downloading: "下载中…",
      error: "下载失败",
      idle: "未下载",
    }[model.state] || model.state;
    addText(heading, "span", stateText, `state ${model.state}`);
    addText(card, "p", model.description);
    if (model.error) addText(card, "p", model.error, "error-text");
    const actions = addText(card, "div", "", "actions");
    addSelect(actions, model.key, model.selected);
    const download = addText(
      actions,
      "button",
      model.state === "downloading" ? "下载中…" : "下载",
      "secondary",
    );
    download.disabled = model.state === "downloading" || model.state === "ready";
    download.addEventListener("click", () => downloadModel(model.key));
    modelsElement.append(card);
  }
}

async function refresh() {
  try {
    render(await bridge.apiGet("semif/models"));
  } catch (error) {
    showNotice(error.message || "读取模型状态失败", true);
  }
}

async function selectModel(key) {
  try {
    await bridge.apiPost("semif/models/select", { model: key });
    showNotice("模型选择已保存，请重载插件后生效。");
    await refresh();
  } catch (error) {
    showNotice(error.message || "保存模型选择失败", true);
  }
}

async function downloadModel(key) {
  try {
    await bridge.apiPost("semif/models/download", { model: key });
    showNotice("已开始下载，页面会自动刷新状态。下载完成后请重载插件。");
    await refresh();
  } catch (error) {
    showNotice(error.message || "开始下载失败", true);
  }
}

await bridge.ready();
await refresh();
setInterval(refresh, 3000);
