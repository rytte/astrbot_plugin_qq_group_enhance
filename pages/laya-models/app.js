const bridge = window.AstrBotPluginPage;
const modelsElement = document.getElementById("models");
const noticeElement = document.getElementById("notice");
const backendButtons = [...document.querySelectorAll("[data-backend]")];
const testForm = document.getElementById("test-form");
const testFields = document.getElementById("test-fields");
const testMode = document.getElementById("test-mode");
const testModel = document.getElementById("test-model");
const testResults = document.getElementById("test-results");
const testError = document.getElementById("test-error");
const testStatus = document.getElementById("test-status");
const testSubmit = document.getElementById("test-submit");
const backends = {
  laya: { label: "Laya", endpoint: "laya/models" },
  semif: { label: "SemIf", endpoint: "semif/models" },
  jev: { label: "Jev" },
};
let currentBackend = "laya";
let settings = null;
let modelSignature = "";
let refreshVersion = 0;
let testing = false;

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

function addButton(actions, label, onClick, disabled = false, className = "") {
  const button = addText(actions, "button", label, className);
  button.type = "button";
  button.disabled = disabled;
  button.addEventListener("click", onClick);
  return button;
}

function fillModelSelect(backend, data) {
  const models = backend === "semif"
    ? [{ key: "custom", label: "自定义模型（插件配置）", state: "ready" }, ...data.models]
    : data.models;
  const signature = JSON.stringify([backend, models.map((model) => [model.key, model.state])]);
  if (signature === modelSignature) return;
  modelSignature = signature;
  const previous = testModel.value;
  testModel.replaceChildren();
  for (const model of models) {
    const option = addText(testModel, "option", `${model.label}${model.state === "ready" ? "" : "（请先下载）"}`);
    option.value = model.key;
    option.disabled = model.state !== "ready";
  }
  const available = models.filter((model) => model.state === "ready");
  const selected = available.find((model) => model.key === previous)
    || available.find((model) => model.key === data.selected)
    || available[0];
  testModel.value = selected?.key || "";
}

function render(backend, data) {
  modelsElement.replaceChildren();
  if (backend === "jev") {
    const card = addText(modelsElement, "article", "", "model-card");
    addText(card, "h2", "Jev 云端模型");
    addText(card, "p", `当前配置：${settings.jev.model}。${settings.jev.configured ? "已配置 API Key，可以直接测试。" : "请在插件配置中填写 API Key 并重载插件。"}`);
    addText(card, "p", "云端模型无需下载。每次测试会调用配置的 Jev 接口，费用按服务商规则计算。");
    return;
  }
  if (backend === "semif") {
    const custom = addText(modelsElement, "article", "", `model-card ${data.selected === "custom" ? "selected" : ""}`);
    addText(custom, "h2", "自定义模型");
    addText(custom, "p", "使用插件配置中的 SemIf 模型目录或 Hugging Face ID，以及固定模型版本。");
    const actions = addText(custom, "div", "", "actions");
    addButton(actions, data.selected === "custom" ? "已保存模型" : "选择模型", () => selectModel(backend, "custom"), data.selected === "custom");
    addButton(actions, "测试模型", () => chooseTestModel("custom"), testing, "secondary");
  }
  for (const model of data.models || []) {
    const card = addText(modelsElement, "article", "", `model-card ${model.selected ? "selected" : ""}`);
    const heading = addText(card, "div", "", "heading");
    const name = addText(heading, "div", "");
    addText(name, "h2", model.label);
    addText(name, "code", model.repo_id || model.key);
    const stateText = { ready: "已下载", downloading: "下载中…", error: "下载失败", idle: "未下载" }[model.state] || model.state;
    addText(heading, "span", stateText, `state ${model.state}`);
    addText(card, "p", model.description);
    if (model.error) addText(card, "p", model.error, "error-text");
    const actions = addText(card, "div", "", "actions");
    addButton(actions, model.selected ? "已保存模型" : "选择模型", () => selectModel(backend, model.key), model.selected);
    addButton(actions, model.state === "downloading" ? "下载中…" : "下载", () => downloadModel(backend, model.key), ["downloading", "ready"].includes(model.state), "secondary");
    addButton(actions, "测试模型", () => chooseTestModel(model.key), model.state !== "ready" || testing, "secondary");
  }
  fillModelSelect(backend, data);
}

function chooseTestModel(key) {
  if (testing) return;
  testModel.value = key;
  document.querySelector(".playground").scrollIntoView({ behavior: "smooth", block: "start" });
  testModel.focus({ preventScroll: true });
}

async function refresh() {
  const backend = currentBackend;
  const version = ++refreshVersion;
  if (backend === "jev") {
    render(backend);
    return;
  }
  try {
    const data = await bridge.apiGet(backends[backend].endpoint);
    if (backend === currentBackend && version === refreshVersion) render(backend, data);
  } catch (error) {
    if (backend === currentBackend && version === refreshVersion) showNotice(error.message || "读取模型状态失败", true);
  }
}

async function selectModel(backend, key) {
  try {
    await bridge.apiPost(`${backends[backend].endpoint}/select`, { model: key });
    showNotice(`${backends[backend].label} 模型选择已保存，重载插件后用于群聊；页面测试可直接使用。`);
    await refresh();
  } catch (error) {
    showNotice(error.message || "保存模型选择失败", true);
  }
}

async function downloadModel(backend, key) {
  try {
    await bridge.apiPost(`${backends[backend].endpoint}/download`, { model: key });
    showNotice("已开始下载，页面会自动刷新状态。下载完成后可直接测试，群聊使用需要重载插件。");
    await refresh();
  } catch (error) {
    showNotice(error.message || "开始下载失败", true);
  }
}

async function switchBackend(backend) {
  if (testing) return;
  currentBackend = backend;
  modelSignature = "";
  testModel.replaceChildren();
  for (const button of backendButtons) button.setAttribute("aria-pressed", String(button.dataset.backend === backend));
  const jev = backend === "jev";
  document.getElementById("local-model-field").hidden = jev;
  document.getElementById("jev-model-field").hidden = !jev;
  document.getElementById("test-jev-model").disabled = !jev;
  testModel.disabled = jev;
  showNotice("");
  testError.hidden = true;
  testStatus.textContent = "";
  testResults.replaceChildren();
  modelsElement.replaceChildren();
  await refresh();
}

function switchMode() {
  const wake = testMode.value === "wake";
  document.getElementById("wake-fields").hidden = !wake;
  document.getElementById("choice-fields").hidden = wake;
  for (const input of document.querySelectorAll("#wake-fields input, #wake-fields textarea")) input.disabled = !wake;
  for (const input of document.querySelectorAll("#choice-fields input, #choice-fields textarea")) input.disabled = wake;
  testError.hidden = true;
  testResults.replaceChildren();
  testStatus.textContent = "";
}

function jsonInput(id, label) {
  try {
    return JSON.parse(document.getElementById(id).value);
  } catch {
    throw new Error(`${label}不是有效的 JSON，请检查双引号、逗号和括号。`);
  }
}

function probabilityRow(parent, label, probability) {
  const row = addText(parent, "div", "", "probability-row");
  const heading = addText(row, "div", "", "probability-label");
  addText(heading, "span", label);
  addText(heading, "strong", `${(probability * 100).toFixed(2)}%`);
  const progress = document.createElement("progress");
  progress.max = 1;
  progress.value = probability;
  progress.setAttribute("aria-label", label);
  row.append(progress);
}

function renderResults(data) {
  testResults.replaceChildren();
  addText(testResults, "p", `${backends[data.backend].label} · ${data.model} · ${(data.elapsed_ms / 1000).toFixed(3)} 秒${data.mode === "wake" ? ` · 阈值 ${(data.threshold * 100).toFixed(0)}%` : ""}`, "result-meta");
  if (data.history_omitted) addText(testResults, "p", `因模型输入上限，本次省略了 ${data.history_omitted} 条最早的历史消息。`, "hint");
  for (const result of data.results) {
    const card = addText(testResults, "article", "", "result-card");
    if (data.mode === "wake") {
      const heading = addText(card, "div", "", "result-heading");
      addText(heading, "strong", result.id);
      addText(heading, "span", result.passed ? "达到唤醒阈值" : "未达到唤醒阈值", `state ${result.passed ? "ready" : "idle"}`);
      addText(card, "p", result.text);
      probabilityRow(card, "对机器人说话的概率", result.probability);
    } else {
      const selected = result.options.find((option) => option.id === result.selected);
      addText(card, "strong", `判定结果：${selected.id}`);
      addText(card, "p", selected.label);
      for (const option of result.options) probabilityRow(card, `${option.id} · ${option.label}`, option.probability);
    }
  }
  const details = addText(testResults, "details", "");
  addText(details, "summary", "查看完整测试结果");
  addText(details, "pre", JSON.stringify(data, null, 2));
}

testForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (testing) return;
  testError.hidden = true;
  testResults.replaceChildren();
  try {
    const payload = { backend: currentBackend, mode: testMode.value };
    payload.model = currentBackend === "jev" ? document.getElementById("test-jev-model").value.trim() : testModel.value;
    if (!payload.model) throw new Error("请先下载并选择一个可测试的模型。");
    if (payload.mode === "wake") {
      payload.bot_id = document.getElementById("test-bot-id").value.trim();
      payload.threshold = Number(document.getElementById("test-threshold").value);
      payload.keywords = document.getElementById("test-keywords").value.split(/[,，\n]/).map((name) => name.trim()).filter(Boolean);
      payload.history = jsonInput("test-history", "历史对话");
      payload.candidates = jsonInput("test-candidates", "待判断消息");
    } else {
      payload.context = document.getElementById("context-json").checked
        ? jsonInput("test-context", "背景") : document.getElementById("test-context").value;
      payload.question = document.getElementById("test-question").value;
      payload.options = document.getElementById("test-options").value.split("\n").map((option) => option.trim()).filter(Boolean);
    }
    testing = true;
    testFields.disabled = true;
    for (const button of backendButtons) button.disabled = true;
    testSubmit.textContent = "测试中…";
    testStatus.textContent = "正在运行模型测试，首次加载可能需要较长时间…";
    renderResults(await bridge.apiPost("semantic/test", payload));
    testStatus.textContent = "测试完成。";
  } catch (error) {
    testStatus.textContent = "";
    testError.textContent = error.message || "模型测试失败";
    testError.hidden = false;
  } finally {
    testing = false;
    testFields.disabled = false;
    for (const button of backendButtons) button.disabled = false;
    testSubmit.textContent = "开始测试";
  }
});

document.getElementById("test-example").addEventListener("click", () => {
  const ids = testMode.value === "wake" ? ["test-history", "test-candidates"] : ["test-context", "test-question", "test-options"];
  for (const id of ids) {
    const input = document.getElementById(id);
    input.value = input.defaultValue;
  }
  if (testMode.value === "wake") document.getElementById("test-bot-id").value = "bot";
  else document.getElementById("context-json").checked = false;
});
testMode.addEventListener("change", switchMode);
for (const button of backendButtons) button.addEventListener("click", () => switchBackend(button.dataset.backend));

try {
  await bridge.ready();
  settings = await bridge.apiGet("semantic/test/settings");
  document.getElementById("test-threshold").value = settings.threshold;
  document.getElementById("test-keywords").value = settings.keywords.join(", ");
  document.getElementById("test-jev-model").value = settings.jev.model;
  switchMode();
  await switchBackend(settings.backend);
  setInterval(() => {
    if (currentBackend !== "jev" && !testing && !document.hidden) refresh();
  }, 3000);
} catch (error) {
  showNotice(error.message || "初始化模型页面失败", true);
  testFields.disabled = true;
}
