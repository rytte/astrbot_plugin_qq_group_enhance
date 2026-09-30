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
let modelData = {};

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

function stateText(state) {
  return { ready: "已下载", downloading: "下载中…", error: "下载失败", idle: "未下载" }[state] || state;
}

function renderModel(parent, backend, model) {
  const card = addText(parent, "article", "", "model-card");
  const heading = addText(card, "div", "", "heading");
  const name = addText(heading, "div", "", "model-name");
  addText(name, "strong", model.label);
  addText(name, "code", model.repo_id || model.key);
  addText(heading, "span", stateText(model.state), `state ${model.state}`);
  if (model.error) addText(card, "p", model.error, "error-text");
  const actions = addText(card, "div", "", "actions");
  addButton(actions, model.state === "downloading" ? "下载中…" : "下载", () => downloadModel(backend, model.key), ["downloading", "ready"].includes(model.state), "secondary");
  addButton(actions, "选择测试", () => chooseTestModel(backend, model.key), model.state !== "ready" || testing);
}

function renderModels() {
  modelsElement.replaceChildren();
  const layaGroup = addText(modelsElement, "section", "", "model-group");
  addText(layaGroup, "h3", "Laya");
  addText(layaGroup, "p", "本地语义唤醒模型", "group-hint");
  for (const model of modelData.laya?.models || []) renderModel(layaGroup, "laya", model);

  const semifGroup = addText(modelsElement, "section", "", "model-group");
  addText(semifGroup, "h3", "SemIf");
  addText(semifGroup, "p", "可下载预设或使用插件配置中的自定义模型", "group-hint");
  const semif = modelData.semif;
  if (semif) {
    const custom = addText(semifGroup, "article", "", "model-card");
    const heading = addText(custom, "div", "", "heading");
    addText(heading, "strong", "自定义模型");
    addText(heading, "span", "可测试", "state ready");
    const actions = addText(custom, "div", "", "actions");
    addButton(actions, "选择测试", () => chooseTestModel("semif", "custom"), testing);
    for (const model of semif.models || []) renderModel(semifGroup, "semif", model);
  }

  const jevGroup = addText(modelsElement, "section", "", "model-group");
  addText(jevGroup, "h3", "Jev");
  addText(jevGroup, "p", `云端模型：${settings.jev.model}。${settings.jev.configured ? "已配置 API Key，可以直接测试。" : "请在插件配置中填写 API Key 并重载插件。"}`, "group-hint");
  const jevCard = addText(jevGroup, "article", "", "model-card compact-card");
  addText(jevCard, "span", "无需下载", "state ready");
  addButton(addText(jevCard, "div", "", "actions"), "选择测试", () => chooseTestModel("jev", settings.jev.model), testing);

  const data = modelData[currentBackend];
  if (currentBackend !== "jev" && data) fillModelSelect(currentBackend, data);
}

function chooseTestModel(backend, key) {
  if (testing) return;
  if (currentBackend !== backend) switchBackend(backend);
  if (backend === "jev") document.getElementById("test-jev-model").value = key;
  else testModel.value = key;
  testStatus.textContent = `${backends[backend].label} · ${key} 已选为本次测试模型。`;
  document.querySelector(".playground").scrollIntoView({ behavior: "smooth", block: "start" });
  (backend === "jev" ? document.getElementById("test-jev-model") : testModel).focus({ preventScroll: true });
}

async function refresh() {
  const version = ++refreshVersion;
  try {
    const [laya, semif] = await Promise.all([
      bridge.apiGet(backends.laya.endpoint),
      bridge.apiGet(backends.semif.endpoint),
    ]);
    if (version !== refreshVersion) return;
    modelData = { laya, semif };
    renderModels();
  } catch (error) {
    if (version === refreshVersion) showNotice(error.message || "读取模型状态失败", true);
  }
}

async function downloadModel(backend, key) {
  try {
    await bridge.apiPost(`${backends[backend].endpoint}/download`, { model: key });
    showNotice("已开始下载，页面会自动刷新状态。下载完成后可在本页选择测试，不会修改插件配置。");
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
  const data = modelData[backend];
  if (backend !== "jev" && data) fillModelSelect(backend, data);
}

function switchMode() {
  const mode = testMode.value;
  const wake = mode === "wake";
  const noul = mode === "noul";
  document.getElementById("wake-fields").hidden = !wake;
  document.getElementById("choice-fields").hidden = wake;
  for (const input of document.querySelectorAll("#wake-fields input, #wake-fields textarea")) input.disabled = !wake;
  for (const input of document.querySelectorAll("#choice-fields input, #choice-fields textarea")) {
    input.disabled = wake || (noul && input.id === "test-options");
  }
  document.getElementById("options-field").hidden = wake || noul;
  const copy = {
    wake: ["使用历史消息和待判断消息，复现群聊唤醒场景。", "候选选项", "唤醒场景会自动生成正反两道评分题。"],
    choice: ["从多个选项中选择概率最高的一项。", "候选选项（每行一项，2～16 项）", "会显示每个选项的概率，并标记最高概率选项。"],
    score: ["对每个选项分别评分，只观察概率，不强制选出赢家。", "评分项（每行一项，2～16 项）", "会保留每个选项的概率，适合比较多个候选。"],
    noul: ["判断一个命题成立的概率；本地模型使用是/否评分，Jev 使用原生 noul。", "候选选项", "Noul 不需要填写选项。"],
  }[mode];
  document.getElementById("test-mode-hint").textContent = copy[0];
  document.getElementById("options-label").textContent = copy[1];
  document.getElementById("options-hint").textContent = copy[2];
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
    } else if (data.mode === "noul") {
      addText(card, "strong", "Noul 概率");
      probabilityRow(card, "命题成立概率", result.probability);
    } else if (data.mode === "score") {
      addText(card, "strong", "Score 结果");
      for (const option of result.options) probabilityRow(card, `${option.id} · ${option.label}`, option.probability);
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
      if (payload.mode !== "noul") {
        payload.options = document.getElementById("test-options").value.split("\n").map((option) => option.trim()).filter(Boolean);
      }
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
  const ids = testMode.value === "wake"
    ? ["test-history", "test-candidates"]
    : ["test-context", "test-question", ...(testMode.value === "noul" ? [] : ["test-options"])];
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
    if (!testing && !document.hidden) refresh();
  }, 3000);
  await refresh();
} catch (error) {
  showNotice(error.message || "初始化模型页面失败", true);
  testFields.disabled = true;
}
