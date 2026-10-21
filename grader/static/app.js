"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const activeRequests = new Set();
  const state = { rubric: null, assignmentDetails: "", gradingGuidance: "", models: [], catalog: [], services: [], runtimeTimer: null, runtimeBudget: 0, runtimeRequest: false, runtimeActions: new Set(), workspaceBusy: false, explanationSettings: null, explanations: [], explanationContext: null, explanationElements: new Map(), defaults: null, result: null, comparison: null, overrides: {}, revision: 0, pageEpoch: 0, busy: false, grading: false, initialized: false, expandedCriteria: new Set() };
  const DEFAULT_VISION_MODEL = "openbmb/minicpm-v4.7:1b";
  state.visionSettings = { model: DEFAULT_VISION_MODEL, auto_detect: true, detailed_crops: true };
  state.vision = { file: null, prepared: null, analysis: null, detected: [], nativeImported: false, busy: false, epoch: 0, controller: null, added: false };
  const modelLetter = (index) => String.fromCharCode(65 + index);
  const selectedEntries = () => state.models.filter((entry) => entry.enabled);
  const currentModels = () => selectedEntries().map((entry) => modelSettings(entry.settings));
  const modelName = (entry) => entry.settings.display_name || entry.settings.model || "Custom local model";
  const SERVICE_IDS = new Set(["clef", "startlux", "d3", "ollama"]);

  const uid = (prefix) => `${prefix}-${crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`}`;
  const clone = (value) => JSON.parse(JSON.stringify(value));
  const number = (value) => Number.isFinite(Number(value)) ? Number(value) : 0;
  const fmt = (value) => value == null ? "—" : number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
  const percent = (value) => value == null ? "—" : `${fmt(number(value) * 100)}%`;
  const maxFor = (criterion) => Math.max(0, ...criterion.levels.map((level) => number(level.points)));
  const node = (tag, className = "", text = "") => { const element = document.createElement(tag); if (className) element.className = className; if (text !== "") element.textContent = text; return element; };
  const namedInput = (tag, className, value, label) => { const element = node(tag, className); element.value = value; element.setAttribute("aria-label", label); element.setAttribute("autocomplete", "off"); return element; };

  function announce(message, error = false) {
    const element = $("global-status");
    element.textContent = message;
    element.className = `global-status${error ? " error" : ""}`;
    element.hidden = !message;
  }

  function status(id, message, error = false) {
    $(id).textContent = message;
    $(id).classList.toggle("error", error);
  }

  function invalidate() {
    state.revision += 1;
    if (state.result || state.comparison) {
      state.result = null;
      state.comparison = null;
      state.overrides = {};
      state.explanations = []; state.explanationContext = null; state.explanationElements.clear();
      $("results-content").replaceChildren();
      $("results-content").hidden = true;
      $("results-empty").hidden = false;
      $("result-actions").hidden = true;
      status("grading-status", "Inputs changed. Grade again to get an up-to-date result.");
    }
    if (state.grading) status("grading-status", "Inputs changed. The pending result will be discarded; grade again when it finishes.");
  }

  function updateRubricMeta() {
    if (!state.rubric) return;
    $("rubric-total").textContent = `${fmt(state.rubric.criteria.reduce((sum, criterion) => sum + maxFor(criterion), 0))} points`;
    $("criteria-count").textContent = `${state.rubric.criteria.length} ${state.rubric.criteria.length === 1 ? "criterion" : "criteria"}`;
  }

  function updateTextMeta() {
    const text = $("submission-text").value;
    const length = Array.from(text).length;
    const words = text.trim() ? text.trim().split(/\s+/u).length : 0;
    const limits = currentModels().map((settings) => settings.max_input_chars);
    const limit = limits.length ? Math.min(...limits) : 200000;
    $("word-count").textContent = `${words.toLocaleString()} ${words === 1 ? "word" : "words"}`;
    $("char-count").textContent = `${length.toLocaleString()} / ${limit.toLocaleString()} characters${new Set(limits).size > 1 ? " · smallest model limit" : ""}`;
    $("char-count").parentElement.classList.toggle("over-limit", length > limit);
  }

  function setBusy(busy) {
    state.busy = busy;
    for (const id of ["grade-button", "probe-model", "import-rubric", "rubric-file", "submission-file", "clear-submission", "forget-workspace"]) $(id).disabled = busy || !state.initialized;
    $("workspace").setAttribute("aria-busy", String(busy));
    renderVisionControls();
    if (state.initialized) renderRuntimes();
  }

  async function api(path, options = {}, timeoutMs = 900000) {
    // All browser requests are same-origin. Model endpoints are only handled by the local backend.
    if (!path.startsWith("/api/")) throw new Error("Only local application requests are allowed.");
    const controller = new AbortController();
    const callerSignal = options.signal;
    const abortFromCaller = () => controller.abort();
    callerSignal?.addEventListener("abort", abortFromCaller, { once: true });
    if (callerSignal?.aborted) controller.abort();
    activeRequests.add(controller);
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(path, { ...options, credentials: "same-origin", cache: "no-store", signal: controller.signal });
      let data;
      try { data = await response.json(); } catch { throw new Error("The local server returned an unreadable response. Check that the app is running."); }
      if (!response.ok) {
        const detail = data.error || data.detail || data.message || `Local request failed (${response.status}).`;
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      return data;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("The local request took too long. Check your model server, then try again.");
      if (error instanceof TypeError) throw new Error("Cannot reach the local application. Make sure its server is still running.");
      throw error;
    } finally { clearTimeout(timer); activeRequests.delete(controller); callerSignal?.removeEventListener("abort", abortFromCaller); }
  }

  function jsonRequest(data) { return { method: "POST", headers: { "Content-Type": "application/json", "X-Margin-Scoring-Version": "3" }, body: JSON.stringify(data) }; }

  function normalizeRubric(value) {
    if (!value || typeof value !== "object" || !Array.isArray(value.criteria)) throw new Error("This JSON file is not a rubric. It must contain a title, instructions, and a criteria array.");
    return {
      title: typeof value.title === "string" ? value.title : "",
      instructions: typeof value.instructions === "string" ? value.instructions : "",
      criteria: value.criteria.map((criterion) => {
        if (!criterion || !Array.isArray(criterion.levels) || criterion.levels.length < 2 || criterion.levels.length > 12) throw new Error("Every criterion must include between 2 and 12 scoring levels.");
        return { id: typeof criterion.id === "string" && criterion.id ? criterion.id : uid("criterion"), name: String(criterion.name || ""), description: String(criterion.description || ""), levels: criterion.levels.map((level) => {
          if (!level || !Number.isFinite(Number(level.points)) || Number(level.points) < 0) throw new Error("All rubric levels need non-negative point values.");
          return { id: typeof level.id === "string" && level.id ? level.id : uid("level"), label: String(level.label || ""), description: String(level.description || ""), points: Number(level.points) };
        }) };
      })
    };
  }

  function validateModelSettings() {
    if (!selectedEntries().length) throw new Error("Enable at least one local model before grading or checking models.");
    if (state.models.length > 8) throw new Error("Configure at most eight local models.");
    const identities = new Set();
    for (const entry of selectedEntries()) {
      validateSettings(entry.settings);
      const identity = JSON.stringify([entry.settings.provider, entry.settings.model, entry.settings.provider === "systemone" ? entry.settings.endpoint : null]);
      if (identities.has(identity)) throw new Error("Enabled entries must use different local model configurations. Disable or remove the duplicate.");
      identities.add(identity);
    }
  }

  function validate() {
    if (!state.rubric) throw new Error("The app is still loading. Please wait a moment.");
    validateModelSettings();
    if (!$("submission-text").value.trim()) { $("submission-text").focus(); throw new Error("Add a submission before grading."); }
    if (Array.from($("submission-text").value).length > 200000) { $("submission-text").focus(); throw new Error("Keep the submission within 200,000 characters. Text is never silently cut."); }
    if (Array.from(state.gradingGuidance).length > 12000) { $("grading-guidance").focus(); throw new Error("Keep instructor grading guidance within 12,000 characters."); }
    if (!state.rubric.title.trim()) { $("rubric-title").focus(); throw new Error("Give your rubric a title before grading."); }
    if (!state.rubric.criteria.length || state.rubric.criteria.length > 32) throw new Error("Your rubric must contain between 1 and 32 criteria.");
    const ids = new Set();
    for (const criterion of state.rubric.criteria) {
      if (ids.has(criterion.id)) throw new Error("Criterion IDs must be unique. Check your imported rubric.");
      ids.add(criterion.id);
      if (!criterion.name.trim()) throw new Error("Give every criterion a name before grading.");
      if (!criterion.description.trim()) throw new Error(`Describe what “${criterion.name}” assesses before grading.`);
      if (criterion.levels.length < 2 || criterion.levels.length > 12) throw new Error(`Use between 2 and 12 scoring levels for “${criterion.name}”.`);
      if (!maxFor(criterion)) throw new Error(`Add a level worth more than zero points to “${criterion.name}”.`);
      const levelIds = new Set();
      for (const level of criterion.levels) {
        if (levelIds.has(level.id)) throw new Error(`Level IDs must be unique within “${criterion.name}”.`);
        levelIds.add(level.id);
        if (!level.label.trim()) throw new Error(`Give every level in “${criterion.name}” a label.`);
        if (!Number.isFinite(level.points) || level.points < 0) throw new Error("Point values must be valid non-negative numbers.");
      }
    }
  }

  function renderRubric() {
    $("rubric-title").value = state.rubric.title;
    $("rubric-instructions").value = state.rubric.instructions;
    const editor = $("criteria-editor");
    editor.replaceChildren();
    $("rubric-empty").hidden = state.rubric.criteria.length > 0;
    if (!state.rubric.criteria.some((criterion) => state.expandedCriteria.has(criterion.id)) && state.rubric.criteria[0]) state.expandedCriteria.add(state.rubric.criteria[0].id);
    state.rubric.criteria.forEach((criterion, criterionIndex) => {
      const card = node("details", "criterion-card");
      card.open = state.expandedCriteria.has(criterion.id);
      const collapseHeader = node("summary", "criterion-summary");
      const summaryName = node("span", "criterion-summary-name", criterion.name || "Untitled criterion");
      const summaryMax = node("span", "criterion-max", `${fmt(maxFor(criterion))} pts max`);
      const arrow = node("span", "criterion-chevron", "⌄"); arrow.setAttribute("aria-hidden", "true");
      collapseHeader.append(node("span", "criterion-number", String(criterionIndex + 1).padStart(2, "0")), summaryName, summaryMax, arrow);
      card.addEventListener("toggle", () => { if (card.open) state.expandedCriteria.add(criterion.id); else state.expandedCriteria.delete(criterion.id); });
      const cardBody = node("div", "criterion-body");
      const head = node("div", "criterion-head");
      const title = namedInput("input", "criterion-name", criterion.name, `Criterion ${criterionIndex + 1} name`);
      title.type = "text";
      title.placeholder = "Criterion name";
      const max = node("span", "criterion-max", `${fmt(maxFor(criterion))} pts max`);
      const remove = node("button", "icon-button", "×");
      remove.type = "button";
      remove.setAttribute("aria-label", `Remove criterion ${criterionIndex + 1}`);
      remove.addEventListener("click", () => { state.rubric.criteria.splice(criterionIndex, 1); invalidate(); renderRubric(); $("add-criterion").focus(); });
      title.addEventListener("input", () => { criterion.name = title.value; summaryName.textContent = criterion.name || "Untitled criterion"; invalidate(); });
      head.append(title, max, remove);
      const description = namedInput("textarea", "criterion-description", criterion.description, `Criterion ${criterionIndex + 1} description`);
      description.rows = 2;
      description.placeholder = "What does this criterion assess?";
      description.addEventListener("input", () => { criterion.description = description.value; invalidate(); });
      const levels = node("div", "levels-list");
      criterion.levels.forEach((level, levelIndex) => {
        const row = node("div", "level-row");
        const label = namedInput("input", "level-label", level.label, `Criterion ${criterionIndex + 1}, level ${levelIndex + 1} label`);
        label.type = "text";
        label.placeholder = "Level label";
        label.addEventListener("input", () => { level.label = label.value; invalidate(); });
        const pointsWrap = node("div", "points-wrap");
        const points = namedInput("input", "level-points", level.points, `Criterion ${criterionIndex + 1}, level ${levelIndex + 1} points`);
        points.type = "number"; points.min = "0"; points.step = "any";
        points.addEventListener("input", () => { level.points = points.value === "" ? 0 : Number(points.value); max.textContent = `${fmt(maxFor(criterion))} pts max`; summaryMax.textContent = max.textContent; updateRubricMeta(); invalidate(); });
        const unit = node("span", "", "pts"); unit.setAttribute("aria-hidden", "true");
        pointsWrap.append(points, unit);
        const removeLevel = node("button", "icon-button", "×");
        removeLevel.type = "button";
        removeLevel.disabled = criterion.levels.length <= 2;
        removeLevel.setAttribute("aria-label", `Remove level ${levelIndex + 1} from criterion ${criterionIndex + 1}`);
        removeLevel.addEventListener("click", () => { criterion.levels.splice(levelIndex, 1); invalidate(); renderRubric(); });
        const descriptor = namedInput("textarea", "level-description", level.description, `Criterion ${criterionIndex + 1}, level ${levelIndex + 1} descriptor`);
        descriptor.rows = 2;
        descriptor.placeholder = "Optional · criterion description supplies guidance";
        descriptor.addEventListener("input", () => { level.description = descriptor.value; invalidate(); });
        row.append(label, pointsWrap, removeLevel, descriptor);
        levels.append(row);
      });
      const addLevel = node("button", "add-level", "+ Add a level");
      addLevel.type = "button";
      addLevel.disabled = criterion.levels.length >= 12;
      addLevel.addEventListener("click", () => { criterion.levels.push({ id: uid("level"), label: "New level", points: 0, description: "" }); invalidate(); renderRubric(); const inputs = editor.children[criterionIndex].querySelectorAll(".level-label"); inputs[inputs.length - 1].focus(); inputs[inputs.length - 1].select(); });
      cardBody.append(head, description, levels, addLevel);
      card.append(collapseHeader, cardBody);
      editor.append(card);
    });
    updateRubricMeta();
    $("add-criterion").disabled = state.rubric.criteria.length >= 32;
  }

  function modelSettings(settings) {
    return { provider: settings.provider, display_name: settings.display_name || "", endpoint: settings.endpoint || "http://127.0.0.1:1234/v1/systemone", model: settings.model || "", confidence_threshold: settings.confidence_threshold ?? 0.65, scoring_mode: settings.scoring_mode || "suggest", max_input_chars: settings.max_input_chars ?? 24000 };
  }

  function validateSettings(settings) {
    if (!settings || !["laya", "systemone"].includes(settings.provider)) throw new Error("Choose Laya or a local System One provider.");
    if (typeof settings.model !== "string" || !settings.model.trim() || settings.model.length > 1000) throw new Error("Every model needs its local model identifier or file path.");
    if (typeof settings.display_name !== "string" || settings.display_name.length > 80) throw new Error("Keep model display names within 80 characters.");
    if (!Number.isFinite(settings.confidence_threshold) || settings.confidence_threshold < 0 || settings.confidence_threshold > 1) throw new Error("Set each review threshold between 0 and 100 percent.");
    if (!["suggest", "strict_review"].includes(settings.scoring_mode)) throw new Error("Choose a supported uncertainty rule for every model.");
    if (!Number.isInteger(settings.max_input_chars) || settings.max_input_chars < 100 || settings.max_input_chars > 200000) throw new Error("Set each character limit to a whole number from 100 to 200,000.");
    if (typeof settings.endpoint !== "string" || settings.endpoint.length > 500) throw new Error("Use a valid local model endpoint.");
    if (settings.provider === "systemone") {
      let url; try { url = new URL(settings.endpoint); } catch { throw new Error("Use a complete loopback URL for each local decision server."); }
      if (!["http:", "https:"].includes(url.protocol) || !["127.0.0.1", "localhost", "[::1]"].includes(url.hostname) || url.username || url.password || url.search || url.hash) throw new Error("Model endpoints must use loopback without credentials, query parameters, or fragments.");
    }
  }

  function normalizeWorkspace(value) {
    const exactKeys = (object, keys) => object && typeof object === "object" && !Array.isArray(object) && Object.keys(object).every((key) => keys.includes(key));
    if (!exactKeys(value, ["schema_version", "models", "explanation_settings", "vision_settings"]) || value.schema_version !== 1 || !Array.isArray(value.models) || value.models.length > 8) throw new Error("Choose a version 1 model configuration with up to eight entries. Rubrics, submissions, and scores do not belong in this file.");
    const ids = new Set();
    const models = value.models.map((entry) => {
      if (!exactKeys(entry, ["id", "enabled", "preset_id", "settings"]) || typeof entry.id !== "string" || !/^[A-Za-z0-9_-]{1,64}$/.test(entry.id) || ids.has(entry.id) || typeof entry.enabled !== "boolean" || ![null, "laya", "clef", "startlux", "d3"].includes(entry.preset_id)) throw new Error("Each configured model needs a unique ID, enabled state, and supported preset or custom connection.");
      ids.add(entry.id);
      if (!exactKeys(entry.settings, ["provider", "display_name", "endpoint", "model", "confidence_threshold", "scoring_mode", "max_input_chars"])) throw new Error("Model settings may contain only connection and scoring options.");
      const settings = modelSettings(entry.settings); validateSettings(settings);
      return { id: entry.id, enabled: entry.enabled, preset_id: entry.preset_id, settings };
    });
    const explanation = value.explanation_settings;
    if (!exactKeys(explanation, ["enabled", "model"]) || typeof explanation.enabled !== "boolean" || typeof explanation.model !== "string" || explanation.model.length > 200 || !/^[A-Za-z0-9][A-Za-z0-9._:/-]*$/.test(explanation.model) || explanation.model.includes("://")) throw new Error("Use the name of an installed local Ollama explanation model.");
    const vision = value.vision_settings === undefined ? { model: DEFAULT_VISION_MODEL } : value.vision_settings;
    if (!exactKeys(vision, ["model", "auto_detect", "detailed_crops"]) || typeof vision.model !== "string" || vision.model.length > 200 || !/^[A-Za-z0-9][A-Za-z0-9._:/-]*$/.test(vision.model) || vision.model.includes("://")) throw new Error("Use the name of an installed local Ollama vision model.");
    if (vision.auto_detect !== undefined && typeof vision.auto_detect !== "boolean") throw new Error("Automatic image detection must be enabled or disabled.");
    if (vision.detailed_crops !== undefined && typeof vision.detailed_crops !== "boolean") throw new Error("Detailed image reading must be enabled or disabled.");
    return { schema_version: 1, models, explanation_settings: { enabled: explanation.enabled, model: explanation.model }, vision_settings: { model: vision.model, auto_detect: vision.auto_detect ?? true, detailed_crops: vision.detailed_crops ?? true } };
  }

  function workspaceSnapshot() {
    // Explicit whitelist: never serialize rubric, assignment, submission, grades, or review notes.
    return normalizeWorkspace({ schema_version: 1, models: state.models.map((entry) => ({ id: entry.id, enabled: entry.enabled, preset_id: entry.preset_id, settings: modelSettings(entry.settings) })), explanation_settings: { enabled: !!state.explanationSettings.enabled, model: state.explanationSettings.model }, vision_settings: { model: state.visionSettings.model, auto_detect: state.visionSettings.auto_detect ?? true, detailed_crops: state.visionSettings.detailed_crops ?? true } });
  }

  function applyWorkspace(value) {
    const workspace = normalizeWorkspace(value);
    state.models = workspace.models;
    state.explanationSettings = workspace.explanation_settings;
    if (state.visionSettings.model !== workspace.vision_settings.model || state.visionSettings.detailed_crops !== workspace.vision_settings.detailed_crops) clearVisionAnalysis();
    state.visionSettings = workspace.vision_settings;
    invalidate(); renderSettings();
  }

  function modelChanged() { invalidate(); updateModelSummary(); status("model-status", ""); status("workspace-status", "Configuration changed in this page. Save explicitly to reuse it."); }

  function updateModelSummary() {
    const count = selectedEntries().length;
    $("provider-summary").textContent = `${count} enabled · ${state.models.length} configured`;
    $("model-count").textContent = `${state.models.length} / 8 configured`;
    if (!state.grading) $("grade-button-label").textContent = count === 1 ? "Grade with selected model" : `Grade with ${count} models`;
    $("grading-model-count").textContent = `${count} selected local ${count === 1 ? "model" : "models"}. One submission.`;
    updateTextMeta();
  }

  function addModel(preset = null) {
    if (state.models.length >= 8) { status("workspace-status", "Eight models are already configured. Remove an entry before adding another.", true); return; }
    const settings = preset ? modelSettings(preset.settings) : modelSettings({ provider: "systemone", model: "", display_name: "Custom local model" });
    state.models.push({ id: uid("model"), enabled: true, preset_id: preset && ["laya", "clef", "startlux", "d3"].includes(preset.id) ? preset.id : null, settings });
    modelChanged(); renderSettings();
    $("model-search").focus();
  }

  function moveModel(id, direction) {
    const index = state.models.findIndex((entry) => entry.id === id), target = index + direction;
    if (index < 0 || target < 0 || target >= state.models.length) return;
    [state.models[index], state.models[target]] = [state.models[target], state.models[index]];
    modelChanged(); renderSettings(); $("move-" + (direction < 0 ? "up-" : "down-") + id).focus();
  }

  function removeModel(id) {
    state.models = state.models.filter((entry) => entry.id !== id);
    modelChanged(); renderSettings(); $("model-search").focus();
  }

  function renderPicker() {
    const query = $("model-search").value.trim().toLocaleLowerCase();
    const results = $("model-search-results"); results.replaceChildren();
    const options = [...state.catalog, ...state.models.filter((entry) => !entry.preset_id).map((entry) => ({ id: entry.id, name: modelName(entry), description: "Copy a custom connection already in this page", settings: entry.settings, service_id: null }))];
    const matches = options.filter((item) => `${item.name} ${item.settings.model} ${item.description}`.toLocaleLowerCase().includes(query));
    for (const item of matches) {
      const row = node("div", "model-search-result"); const text = node("div");
      text.append(node("strong", "", item.name), node("p", "field-help", item.description || "Local decision model"));
      const service = state.services.find((candidate) => candidate.id === item.service_id);
      text.append(node("p", "catalog-state", item.service_id ? (service ? `${service.status}${service.ready ? " · ready" : ""}` : "Local server · check status below") : item.settings.provider === "laya" ? "Embedded · runs inside this app" : "Custom server · started manually"));
      const button = node("button", "small-button", "Add model"); button.type = "button"; button.disabled = state.models.length >= 8;
      button.setAttribute("aria-label", `Add ${item.name}`); button.addEventListener("click", () => addModel(item)); row.append(text, button); results.append(row);
    }
    if (!matches.length) results.append(node("p", "field-help", "No local presets or configured models match. Add a custom local server below."));
    $("add-custom-model").disabled = state.models.length >= 8;
  }

  function renderSettings() {
    const list = $("model-list"); list.replaceChildren();
    if (!state.models.length) list.append(node("p", "notice", "No models configured. Search the local catalog above to add a model before grading."));
    state.models.forEach((entry, index) => {
      const card = node("section", "model-config-card"); card.setAttribute("aria-label", `${modelName(entry)} configuration`);
      const header = node("div", "model-entry-header");
      const enabledLabel = node("label", "check-label"); const enabled = node("input"); enabled.type = "checkbox"; enabled.id = "enabled-" + entry.id; enabled.checked = entry.enabled;
      const name = node("span", "", modelName(entry)); enabledLabel.append(enabled, name);
      enabled.addEventListener("change", () => { entry.enabled = enabled.checked; modelChanged(); });
      const controls = node("div", "model-entry-actions");
      for (const [direction, symbol, label] of [[-1, "↑", "up"], [1, "↓", "down"]]) {
        const button = node("button", "icon-button", symbol); button.type = "button"; button.id = `move-${label}-${entry.id}`; button.setAttribute("aria-label", `Move ${modelName(entry)} ${label}`); button.disabled = direction < 0 ? index === 0 : index === state.models.length - 1; button.addEventListener("click", () => moveModel(entry.id, direction)); controls.append(button);
      }
      const remove = node("button", "icon-button", "×"); remove.type = "button"; remove.id = "remove-" + entry.id; remove.setAttribute("aria-label", `Remove ${modelName(entry)}`); remove.addEventListener("click", () => removeModel(entry.id)); controls.append(remove); header.append(enabledLabel, controls); card.append(header);
      const details = node("details", "model-advanced"); details.open = !entry.settings.model; details.append(node("summary", "", "Connection and scoring settings")); const fields = node("div", "model-fields");
      const field = (key, labelText, type = "text", options = null, help = "") => {
        const wrap = node("div", "field"), label = node("label", "", labelText); const id = `${key}-${entry.id}`; label.htmlFor = id;
        const input = node(options ? "select" : "input"); input.id = id; if (!options) input.type = type; input.setAttribute("autocomplete", "off"); input.spellcheck = false;
        if (options) for (const [value, text] of options) { const option = node("option", "", text); option.value = value; input.append(option); }
        input.value = key === "confidence_threshold" ? entry.settings[key] * 100 : entry.settings[key];
        if (key === "display_name") input.maxLength = 80;
        if (key === "confidence_threshold") { input.min = 0; input.max = 100; input.step = "any"; }
        if (key === "max_input_chars") { input.min = 100; input.max = 200000; input.step = 1; }
        input.addEventListener(options ? "change" : "input", () => {
          entry.settings[key] = key === "confidence_threshold" ? Number(input.value) / 100 : key === "max_input_chars" ? Number(input.value) : input.value;
          if (key === "provider" && input.value === "systemone" && entry.settings.model === "models/laya-multilingual") { entry.settings.model = ""; $("model-" + entry.id).value = ""; }
          endpointWrap.hidden = entry.settings.provider === "laya";
          name.textContent = modelName(entry); modelChanged();
        });
        wrap.append(label, input); if (help) wrap.append(node("p", "field-help", help)); fields.append(wrap); return wrap;
      };
      field("display_name", "Display name · optional");
      field("provider", "Inference provider", "text", [["laya", "Laya · embedded offline model"], ["systemone", "System One · local decision server"]]);
      const endpointWrap = field("endpoint", "Local endpoint", "url", null, "Loopback only. Custom servers must support System One decisions."); endpointWrap.hidden = entry.settings.provider === "laya";
      field("model", "Model identifier or local path", "text", null, "Required. Use cached Laya weights or the exact identifier served locally. No automatic downloads.");
      field("scoring_mode", "When this model is uncertain", "text", [["suggest", "Show provisional points"], ["strict_review", "Withhold uncertain scores"]]);
      field("confidence_threshold", "Review threshold (%)", "number", null, "A review signal, not an accuracy guarantee.");
      field("max_input_chars", "Maximum input characters", "number", null, "This model rejects longer text; other selected models may still return results. Text is never cut.");
      details.append(fields); card.append(details);
      const preset = state.catalog.find((item) => item.id === entry.preset_id);
      const guide = node("details", "model-guide"); guide.append(node("summary", "", "Setup guide"));
      guide.append(node("p", "field-help", preset?.description || "Start a compatible decision server locally, then enter its loopback endpoint and exact model identifier. Custom connections are managed in their original terminal."));
      const link = node("a", "text-link", "Open local setup guide ↗"); link.href = `/setup#${entry.preset_id || "custom-models"}`; link.target = "_blank"; link.rel = "noopener"; guide.append(link); card.append(guide); list.append(card);
    });
    $("explanation-enabled").checked = !!state.explanationSettings?.enabled;
    $("explanation-model").value = state.explanationSettings?.model || "qwen38-27b:q4_k_m";
    $("vision-model").value = state.visionSettings.model;
    $("vision-auto-detect").checked = state.visionSettings.auto_detect ?? true;
    $("vision-detailed-crops").checked = state.visionSettings.detailed_crops ?? true;
    updateModelSummary(); renderPicker(); renderRuntimes();
  }

  function renderRuntimes() {
    const list = $("runtime-list"); list.replaceChildren();
    for (const service of state.services.filter((item) => SERVICE_IDS.has(item.id))) {
      const row = node("div", "runtime-row"); const text = node("div");
      text.append(node("strong", "", service.name), node("p", "field-help", `${service.status}${service.ready ? " · ready" : ""}. ${service.message || ""}`));
      if (service.status === "external") text.append(node("p", "field-help", "Started outside this app. Use its original terminal to stop it."));
      const actions = node("div", "runtime-actions");
      for (const action of ["start", "stop"]) {
        const button = node("button", "small-button", action === "start" ? "Start" : "Stop"); button.type = "button"; button.id = `${action}-runtime-${service.id}`;
        button.disabled = state.runtimeActions.has(service.id) || state.busy || !service[`can_${action}`] || (action === "stop" && !service.owned);
        button.setAttribute("aria-label", `${action === "start" ? "Start" : "Stop"} ${service.name}`); button.addEventListener("click", () => runtimeAction(service.id, action)); actions.append(button);
      }
      row.append(text, actions); list.append(row);
    }
    if (!state.services.length) list.append(node("p", "field-help", "Expand this workspace or refresh to check local server status. Laya runs inside this app."));
  }

  function stopRuntimePolling() { clearTimeout(state.runtimeTimer); state.runtimeTimer = null; state.runtimeBudget = 0; }
  async function refreshRuntimes() {
    if (state.runtimeRequest) return;
    const epoch = state.pageEpoch; state.runtimeRequest = true;
    try {
      const response = await api("/api/runtimes", {}, 15000);
      if (epoch !== state.pageEpoch) return;
      state.services = Array.isArray(response.services) ? response.services.filter((service) => SERVICE_IDS.has(service.id)) : [];
      renderRuntimes(); renderPicker(); status("runtime-status", "");
    } catch (error) { if (epoch === state.pageEpoch) status("runtime-status", error.message, true); }
    finally {
      state.runtimeRequest = false;
      if (epoch === state.pageEpoch && $("model-settings").open && state.runtimeBudget > 0) { state.runtimeBudget -= 1; state.runtimeTimer = setTimeout(refreshRuntimes, 5000); }
    }
  }
  function startRuntimePolling() { stopRuntimePolling(); state.runtimeBudget = 60; void refreshRuntimes(); }
  async function runtimeAction(id, action) {
    const service = state.services.find((item) => item.id === id);
    if (!SERVICE_IDS.has(id) || !["start", "stop"].includes(action) || state.runtimeActions.has(id) || state.busy || !service?.[`can_${action}`] || (action === "stop" && !service.owned)) return;
    const epoch = state.pageEpoch; state.runtimeActions.add(id); renderRuntimes(); status("runtime-status", `${action === "start" ? "Starting" : "Stopping"} ${service.name}…`);
    try {
      const response = await api(`/api/runtimes/${id}/${action}`, { method: "POST" }, 180000);
      if (epoch !== state.pageEpoch) return;
      if (response.service?.id === id) state.services = state.services.map((item) => item.id === id ? response.service : item);
      status("runtime-status", response.service?.message || "Local server action completed.");
    } catch (error) { if (epoch === state.pageEpoch) status("runtime-status", error.message, true); }
    finally { state.runtimeActions.delete(id); if (epoch === state.pageEpoch) { renderRuntimes(); renderPicker(); startRuntimePolling(); } }
  }

  async function workspaceAction(action, file = null) {
    if (state.workspaceBusy) return;
    const revision = state.revision, epoch = state.pageEpoch;
    const visionSettings = JSON.stringify(state.visionSettings);
    state.workspaceBusy = true;
    for (const id of ["save-workspace", "load-workspace", "import-workspace"]) $(id).disabled = true;
    status("workspace-status", action === "save" ? "Saving model configuration only…" : "Loading model configuration…");
    try {
      let response;
      if (action === "save") response = await api("/api/workspace", { ...jsonRequest(workspaceSnapshot()), method: "PUT" }, 20000);
      else if (action === "load") response = await api("/api/workspace", {}, 20000);
      else {
        if (!file || file.size > 64 * 1024) throw new Error("Choose a model configuration JSON file no larger than 64 KB.");
        const raw = await file.text();
        normalizeWorkspace(JSON.parse(raw));
        if (revision !== state.revision || epoch !== state.pageEpoch || visionSettings !== JSON.stringify(state.visionSettings)) return;
        response = await api("/api/workspace/validate", { ...jsonRequest(null), body: raw }, 20000);
      }
      if (epoch !== state.pageEpoch) return;
      if (action === "save") status("workspace-status", revision === state.revision && visionSettings === JSON.stringify(state.visionSettings) ? "Model, explanation, and vision settings saved on this computer. No rubric, assignment, submission, images, or scores were saved." : "The earlier model configuration was saved. Your newer changes remain unsaved in this page.");
      else if (revision === state.revision && visionSettings === JSON.stringify(state.visionSettings)) { applyWorkspace(response.workspace); status("workspace-status", action === "load" ? (response.saved ? "Saved model configuration loaded." : "Default model configuration loaded; no saved configuration exists.") : "Configuration imported into this page. Save explicitly to reuse it. Student work and rubric were kept."); }
      else status("workspace-status", "Configuration changed while loading. The response was discarded; load again if needed.");
    } catch (error) { if (epoch === state.pageEpoch) status("workspace-status", error instanceof SyntaxError ? "This file is not valid model configuration JSON." : error.message, true); }
    finally { state.workspaceBusy = false; for (const id of ["save-workspace", "load-workspace", "import-workspace"]) $(id).disabled = false; }
  }

  function showWarnings(id, warnings) {
    const element = $(id);
    element.replaceChildren();
    for (const warning of warnings || []) element.append(node("p", "", String(warning)));
    element.hidden = !warnings?.length;
  }

  function renderVisionControls() {
    const vision = state.vision;
    const unavailable = state.busy || !state.initialized;
    $("vision-file").disabled = unavailable;
    $("vision-analyze").disabled = unavailable || vision.busy || !vision.prepared;
    $("vision-clear").disabled = !vision.file && !vision.prepared && !vision.busy;
    $("vision-show-page").disabled = unavailable || vision.busy || !vision.file;
    $("vision-detected-select").disabled = unavailable || vision.busy || !vision.file;
    $("vision-index").disabled = unavailable || vision.busy;
    $("vision-image-index").disabled = unavailable || vision.busy;
    $("vision-reviewed").disabled = vision.busy || !vision.analysis || vision.added;
    $("vision-append").disabled = unavailable || vision.busy || !vision.analysis || vision.added || !$("vision-reviewed").checked || ![visionNativeAppendText(), $("vision-transcription").value, $("vision-description").value].some((text) => text.trim());
    $("grade-button").disabled = unavailable || vision.busy;
    $("vision-panel").setAttribute("aria-busy", String(vision.busy));
  }

  function clearVisionAnalysis() {
    const vision = state.vision;
    vision.epoch += 1;
    vision.controller?.abort();
    vision.controller = null;
    vision.busy = false;
    vision.analysis = null;
    vision.added = false;
    $("vision-transcription").value = "";
    $("vision-description").value = "";
    $("vision-reviewed").checked = false;
    $("vision-analysis-model").textContent = "";
    $("vision-analysis").hidden = true;
    $("vision-views").hidden = true;
    $("vision-view-results").replaceChildren();
    $("vision-transcription-field").hidden = false;
    showWarnings("vision-uncertainties", []);
    status("vision-status", "");
    renderVisionControls();
  }

  function clearVision() {
    clearVisionAnalysis();
    state.vision.file = null;
    state.vision.prepared = null;
    state.vision.detected = [];
    state.vision.nativeImported = false;
    $("vision-file").value = "";
    $("vision-preview").hidden = true;
    $("vision-image").removeAttribute("src");
    $("vision-image").alt = "";
    $("vision-source").textContent = "";
    $("vision-index").value = "1";
    $("vision-index").max = "1";
    $("vision-count").textContent = "";
    $("vision-image-controls").hidden = true;
    $("vision-image-index").replaceChildren();
    $("vision-native").hidden = true;
    $("vision-native-text").value = "";
    $("vision-native-note").textContent = "";
    $("vision-detected").hidden = true;
    $("vision-detected-select").replaceChildren();
    $("vision-detected-summary").textContent = "";
    showWarnings("vision-warnings", []);
    renderVisionControls();
  }

  function visionItemKey(item) {
    return item.image_index === undefined ? String(item.index) : `${item.index}:${item.image_index}`;
  }

  function visionUsesNativeText() {
    return state.vision.prepared?.kind === "pdf_vectors" && !!state.vision.prepared.native_text?.trim();
  }

  function visionNativeAppendText() {
    return visionUsesNativeText() && !state.vision.nativeImported ? state.vision.prepared.native_text : "";
  }

  async function prepareVision(file, index = 1, detected = [], imageIndex = undefined, nativeImported = false) {
    if (!file || state.busy || !state.initialized) return;
    clearVision();
    if (file.size > 15 * 1024 * 1024) { status("vision-status", "Choose one image or document no larger than 15 MB.", true); return; }
    if (!/\.(png|jpe?g|webp|pdf|docx)$/i.test(file.name)) { status("vision-status", "Choose a PNG, JPEG, WebP, PDF, or Word (.docx) file.", true); return; }
    if (!Number.isInteger(index) || index < 1) { status("vision-status", "Choose a whole page or image number starting at 1.", true); return; }
    if (imageIndex !== undefined && (!Number.isInteger(imageIndex) || imageIndex < 0 || imageIndex > 300)) { status("vision-status", "Choose an embedded image or the page layout.", true); return; }
    const vision = state.vision;
    vision.file = file;
    vision.nativeImported = nativeImported;
    vision.detected = detected.filter((item) => item && Number.isInteger(item.index) && item.index >= 1 && item.index <= 300 && typeof item.label === "string" && (item.image_index === undefined || (Number.isInteger(item.image_index) && item.image_index >= 0 && item.image_index <= 300))).slice(0, 300).map((item) => ({ index: item.index, label: item.label, ...(item.image_index === undefined ? {} : { image_index: item.image_index }) }));
    if (vision.detected.length) {
      for (const item of vision.detected) { const option = node("option", "", item.label); option.value = visionItemKey(item); $("vision-detected-select").append(option); }
      $("vision-detected-select").value = visionItemKey({ index, image_index: imageIndex });
      $("vision-detected-summary").textContent = `${vision.detected.length} visual ${vision.detected.length === 1 ? "item" : "items"} detected in this submission. Check each relevant item; detection can miss some content.`;
      $("vision-detected").hidden = false;
    }
    vision.busy = true;
    vision.controller = new AbortController();
    const epoch = vision.epoch;
    const pageEpoch = state.pageEpoch;
    renderVisionControls();
    status("vision-status", "Preparing a local image preview…");
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("index", String(index));
      if (imageIndex !== undefined) form.append("image_index", String(imageIndex));
      const data = await api("/api/vision/prepare", { method: "POST", body: form, signal: vision.controller.signal }, 120000);
      if (epoch !== vision.epoch || pageEpoch !== state.pageEpoch) return;
      if (typeof data.image_data_url !== "string" || data.image_data_url.length > 24 * 1024 * 1024 || !/^data:image\/(?:png|jpeg|webp);base64,[A-Za-z0-9+/]+={0,2}$/.test(data.image_data_url) || !Number.isInteger(data.index) || !Number.isInteger(data.count) || data.index !== index || data.count < index || !Number.isInteger(data.width) || !Number.isInteger(data.height) || data.width < 1 || data.height < 1) throw new Error("The local server returned an invalid image preview.");
      if (data.native_text !== undefined && (typeof data.native_text !== "string" || data.native_text.length > 200000)) throw new Error("The local server returned invalid native PDF text.");
      if (data.image_count !== undefined && (!Number.isInteger(data.image_count) || data.image_count < 0 || data.image_count > 300 || !Number.isInteger(data.image_index) || data.image_index < 0 || data.image_index > data.image_count || (imageIndex !== undefined && data.image_index !== imageIndex))) throw new Error("The local server returned an invalid embedded image selection.");
      vision.prepared = data;
      $("vision-image").src = data.image_data_url;
      $("vision-image").alt = "Local image preview. Review the original visual content against the extracted text.";
      $("vision-source").textContent = String(data.source_label || "Image") + ` · ${data.width} × ${data.height} pixels`;
      $("vision-index").value = String(data.index);
      $("vision-index").max = String(data.count);
      $("vision-count").textContent = `of ${data.count}`;
      $("vision-preview").hidden = false;
      if (Number.isInteger(data.image_count) && data.image_count > 0) {
        const pageOption = node("option", "", "Page layout"); pageOption.value = "0"; $("vision-image-index").append(pageOption);
        for (let image = 1; image <= data.image_count; image += 1) { const option = node("option", "", `Embedded image ${image} of ${data.image_count}`); option.value = String(image); $("vision-image-index").append(option); }
        $("vision-image-index").value = String(data.image_index);
        $("vision-image-controls").hidden = false;
      }
      const nativeText = data.native_text || "";
      $("vision-native").hidden = !nativeText;
      $("vision-native-text").value = nativeText;
      $("vision-native-note").textContent = nativeImported ? "Preserved in the submission text. Vision analysis does not rewrite or append this text." : "Extracted directly from this PDF page. When reading the page layout, adding reviewed content includes this text without rewriting it.";
      if (data.image_index !== undefined) $("vision-detected-select").value = visionItemKey({ index, image_index: data.image_index });
      showWarnings("vision-warnings", data.warnings);
      status("vision-status", "Preview ready. Analyze locally, then check the transcription and visual description.");
    } catch (error) { if (epoch === vision.epoch && pageEpoch === state.pageEpoch) status("vision-status", error.message, true); }
    finally { if (epoch === vision.epoch && pageEpoch === state.pageEpoch) { vision.busy = false; vision.controller = null; renderVisionControls(); } }
  }

  async function analyzeVision() {
    const vision = state.vision;
    if (state.busy || vision.busy || !vision.prepared || !state.initialized) return;
    const model = state.visionSettings.model;
    const detailedCrops = state.visionSettings.detailed_crops ?? true;
    const transcribe = !visionUsesNativeText();
    if (typeof model !== "string" || model.length > 200 || !/^[A-Za-z0-9][A-Za-z0-9._:/-]*$/.test(model) || model.includes("://")) { status("vision-status", "Enter an installed local vision model name.", true); return; }
    clearVisionAnalysis();
    vision.busy = true;
    vision.controller = new AbortController();
    const epoch = vision.epoch;
    const pageEpoch = state.pageEpoch;
    renderVisionControls();
    status("vision-status", detailedCrops ? "Reading the overview and detailed crops locally, one view at a time… This takes longer than a single image." : "Analyzing the image on this computer… The first model load can take longer.");
    try {
      const data = await api("/api/vision/analyze", { ...jsonRequest({ image_data_url: vision.prepared.image_data_url, model, detailed_crops: detailedCrops, transcribe }), signal: vision.controller.signal });
      if (epoch !== vision.epoch || pageEpoch !== state.pageEpoch || model !== state.visionSettings.model || detailedCrops !== (state.visionSettings.detailed_crops ?? true)) return;
      if (typeof data.transcription !== "string" || typeof data.description !== "string" || !Array.isArray(data.uncertainties) || data.uncertainties.some((item) => typeof item !== "string")) throw new Error("The local vision model returned an unreadable analysis. No submission text or grades were changed.");
      const views = data.views ?? [];
      if (!Array.isArray(views) || views.length > 10 || views.some((view) => !view || typeof view.label !== "string" || typeof view.transcription !== "string" || typeof view.description !== "string" || !Array.isArray(view.uncertainties) || view.uncertainties.some((item) => typeof item !== "string") || !Array.isArray(view.bounds) || view.bounds.length !== 4 || view.bounds.some((value) => !Number.isFinite(value) || value < 0) || view.bounds[0] >= view.bounds[2] || view.bounds[1] >= view.bounds[3] || view.bounds[2] > vision.prepared.width || view.bounds[3] > vision.prepared.height)) throw new Error("The local vision model returned unreadable crop details. No submission text or grades were changed.");
      const incompleteViews = data.incomplete_views ?? 0;
      if (!Number.isInteger(incompleteViews) || incompleteViews < 0 || incompleteViews > views.length || views.some((view) => view.status !== undefined && !["complete", "incomplete"].includes(view.status)) || views.filter((view) => view.status === "incomplete").length !== incompleteViews || views.some((view) => view.status === "incomplete" && (view.transcription || view.description))) throw new Error("The local vision model returned invalid partial results. No submission text or grades were changed.");
      vision.analysis = { model, uncertainties: data.uncertainties };
      $("vision-transcription").value = transcribe ? data.transcription : "";
      $("vision-transcription-field").hidden = !transcribe;
      for (const view of views) {
        const details = node("details", "vision-view");
        details.append(node("summary", "", `${view.label}${view.status === "incomplete" ? " · Could not read this view" : ""}`));
        details.append(node("p", "field-help", `Source region: ${view.bounds.join(", ")} (left, top, right, bottom).`));
        if (transcribe && view.transcription) { details.append(node("p", "vision-view-label", "Transcribed text")); details.append(node("pre", "vision-view-text", view.transcription)); }
        if (view.description) { details.append(node("p", "vision-view-label", "Visual description")); details.append(node("pre", "vision-view-text", view.description)); }
        for (const uncertainty of view.uncertainties) details.append(node("p", "field-help", uncertainty));
        $("vision-view-results").append(details);
      }
      $("vision-views").hidden = !views.length;
      $("vision-description").value = data.description;
      $("vision-analysis-model").textContent = `${model} · ${views.length || 1} local ${views.length === 1 || !views.length ? "view" : "views"} · Draft extraction, not a grade`;
      showWarnings("vision-uncertainties", data.uncertainties);
      $("vision-analysis").hidden = false;
      status("vision-status", incompleteViews ? `Partial reading: ${incompleteViews} of ${views.length} views could not be read. Completed views are available below. Unread regions are not missing student work; check the original before adding evidence.` : "Check labels, values, arrows, and relationships against the preview. Correct any mistakes before adding the text.", incompleteViews > 0);
    } catch (error) { if (epoch === vision.epoch && pageEpoch === state.pageEpoch) status("vision-status", error.message, true); }
    finally { if (epoch === vision.epoch && pageEpoch === state.pageEpoch) { vision.busy = false; vision.controller = null; renderVisionControls(); } }
  }

  function appendVision() {
    const vision = state.vision;
    if (state.busy || vision.busy || !vision.analysis || !vision.prepared || vision.added || !$("vision-reviewed").checked) return;
    const transcription = visionUsesNativeText() ? "" : $("vision-transcription").value.trim();
    const nativeText = visionNativeAppendText();
    const description = $("vision-description").value.trim();
    if (!transcription && !description && !nativeText.trim()) { status("vision-status", "There is no reviewed content to add. Recognition failure does not mean the student's answer is missing.", true); return; }
    const source = String(vision.prepared.source_label || "Image").replace(/[\r\n]/g, " ");
    const parts = [`[Reviewed image evidence: ${source}]`];
    if (nativeText) parts.push(`Native PDF text (verbatim):\n${nativeText}`);
    if (transcription) parts.push(`Transcribed text (reviewed):\n${transcription}`);
    if (description) parts.push(`AI-assisted visual description (reviewed):\n${description}`);
    if (vision.analysis.uncertainties.length) parts.push(`Recognition uncertainties to consider:\n${vision.analysis.uncertainties.join("\n")}`);
    parts.push("[End reviewed image evidence]");
    const text = [$("submission-text").value, parts.join("\n\n")].filter(Boolean).join("\n\n");
    if (Array.from(text).length > 200000) { status("vision-status", "Adding this content would exceed the 200,000-character submission limit. Shorten the reviewed text first.", true); return; }
    $("submission-text").value = text;
    vision.added = true;
    $("vision-reviewed").checked = false;
    invalidate(); updateTextMeta(); renderVisionControls();
    status("vision-status", "Reviewed image content added. Check the combined submission, then grade again when ready.");
  }

  async function extractFile(file, kind) {
    if (!file || state.busy) return;
    if (file.size > 15 * 1024 * 1024) { announce("This document exceeds the 15 MB upload limit.", true); return; }
    if (kind === "submission" && /\.(png|jpe?g|webp)$/i.test(file.name)) {
      $("vision-panel").open = true;
      return prepareVision(file);
    }
    if (kind === "submission") clearVision();
    announce("");
    setBusy(true);
    const originalRevision = state.revision;
    const pageEpoch = state.pageEpoch;
    const detectVisuals = kind === "submission" && (state.visionSettings.auto_detect ?? true) && /\.(pdf|docx)$/i.test(file.name);
    let detected = [], acceptedRevision = null;
    status(kind === "submission" ? "grading-status" : "model-status", `Reading ${file.name} locally…`);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("detect_visuals", String(detectVisuals));
      const data = await api("/api/extract", { method: "POST", body: form }, 120000);
      if (state.revision !== originalRevision) throw new Error("Your workspace changed while the document was being read. Import it again when you are ready to replace the text.");
      if (kind === "submission") {
        $("submission-text").value = data.text || "";
        $("submission-name").value = data.filename || file.name;
        showWarnings("extraction-warnings", data.warnings);
        updateTextMeta();
        invalidate();
        acceptedRevision = state.revision;
        if (detectVisuals && Array.isArray(data.visuals)) detected = data.visuals.filter((item) => item && Number.isInteger(item.index) && item.index >= 1 && item.index <= 300 && typeof item.label === "string");
        status("grading-status", `Text extracted from ${file.name}${data.pages ? ` · ${data.pages} ${data.pages === 1 ? "page" : "pages"}` : ""}.${detected.length ? ` ${detected.length} visual ${detected.length === 1 ? "item" : "items"} detected for review below.` : ""} Check it before grading.`);
      } else {
        state.rubric.instructions = data.text || "";
        invalidate();
        renderRubric();
        announce(`Imported ${file.name} into grading instructions. Define or update the scored criteria below.${data.warnings?.length ? ` ${data.warnings.join(" ")}` : ""}`);
        status("model-status", "");
      }
    } catch (error) { if (state.pageEpoch === pageEpoch) { announce(error.message, true); status(kind === "submission" ? "grading-status" : "model-status", ""); } }
    finally { setBusy(false); }
    // Reuse the same in-memory File, after text extraction has released its busy
    // state. A changed/cleared submission must never start a stale preview.
    if (detected.length && acceptedRevision === state.revision && pageEpoch === state.pageEpoch && (state.visionSettings.auto_detect ?? true)) {
      $("vision-panel").open = true;
      await prepareVision(file, detected[0].index, detected, detected[0].image_index, true);
    }
  }

  function download(filename, content, type) {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const link = node("a");
    link.href = url; link.download = filename;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function safeFilename(value) { return String(value || "margin").replace(/[^a-zA-Z0-9_\-\u4e00-\u9fff]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 70) || "margin"; }

  function adjustedPoints(criterion) {
    const override = state.overrides[criterion.id];
    return override?.points ?? null;
  }

  function refreshTotal() {
    if (!state.result) return;
    const total = state.result.criteria.reduce((sum, criterion) => sum + (adjustedPoints(criterion) ?? 0), 0);
    const confirmed = state.result.criteria.filter((criterion) => state.overrides[criterion.id]?.points !== null && state.overrides[criterion.id]?.points !== undefined).length;
    const complete = confirmed === state.result.criteria.length;
    $("final-total").textContent = complete ? fmt(total) : "—";
    $("final-percent").textContent = complete ? "Instructor-confirmed score" : "Your independent review";
    $("original-total").textContent = complete ? `${fmt(total / state.result.max_points * 100)}% · All ${confirmed} criterion scores confirmed by you.` : `${confirmed} / ${state.result.criteria.length} scores confirmed. Enter your own score for every criterion to complete the grade.`;
  }

  function modelDecision(entry, criterionId) {
    const block = node("div", "model-decision-block");
    block.append(node("p", "model-label", entry.label));
    const criterion = entry.ok ? entry.result.criteria.find((item) => item.id === criterionId) : null;
    if (!criterion) { block.append(node("p", "model-unavailable", entry.error || "No decision available.")); return block; }
    const withheld = criterion.points == null;
    const top = criterion.top_choice || { label: criterion.level_label, points: criterion.points, description: criterion.description, probability: criterion.confidence };
    if (withheld) {
      block.append(node("h4", "abstention-title", "Review required"));
      block.append(node("p", "abstention-reason", criterion.review_reason || "The model is uncertain. No score was assigned."));
      if (top.label) block.append(node("p", "top-preference", `Top preference: ${top.label} · ${percent(top.probability)}`));
      if (top.points != null) block.append(node("p", "top-preference-note", `This rubric level is worth ${fmt(top.points)} points; this is not an assigned score.`));
    } else {
      const score = node("div", "criterion-score", fmt(criterion.points)); score.append(node("span", "", `/ ${fmt(criterion.max_points)}`)); block.append(score);
      if (criterion.needs_review) block.append(node("p", "provisional-label", "Provisional suggestion · review recommended"));
      block.append(node("p", "selected-level", criterion.needs_review ? `Top preference: ${criterion.level_label}` : criterion.level_label));
      block.append(node("p", "selected-description", criterion.description || "Uses the criterion guidance above."));
      const decision = node("div", "model-decision"); decision.append(node("span", "", "Level probability"), node("span", "", percent(criterion.confidence))); block.append(decision);
      if (criterion.needs_review && criterion.review_reason) block.append(node("p", "provisional-reason", criterion.review_reason));
    }
    const distribution = node("details", "probabilities"); distribution.append(node("summary", "", "All level probabilities"));
    const table = node("table", "probability-table");
    table.append(node("caption", "visually-hidden", `${entry.label}: model probabilities`));
    const tableHead = node("thead"); const headerRow = node("tr"); for (const label of ["Level", "Points", "Probability"]) { const th = node("th", "", label); th.scope = "col"; headerRow.append(th); } tableHead.append(headerRow); table.append(tableHead);
    const body = node("tbody");
    for (const probability of criterion.probabilities || []) { const row = node("tr"); row.append(node("td", "", probability.label), node("td", "", fmt(probability.points)), node("td", "", percent(probability.probability))); body.append(row); }
    table.append(body); distribution.append(table); block.append(distribution);
    return block;
  }

  function prepareExplanations(comparison) {
    state.explanations = []; state.explanationContext = null; state.explanationElements.clear();
    if (!state.explanationSettings?.enabled || (comparison.results.length < 2 || comparison.results.length > 8) || !comparison.results.every((entry) => entry.ok)) return;
    for (const criterion of state.result.rubric.criteria) {
      const rows = comparison.results.map((entry) => entry.result.criteria.find((row) => row.id === criterion.id));
      if (!rows.every((row) => row?.level_id && row.points != null)) continue;
      const unanimous = rows.every((row) => row.level_id === rows[0].level_id);
      if (unanimous && rows[0].points >= maxFor(criterion)) continue;
      state.explanations.push({ criterion_id: criterion.id, mode: unanimous ? "agreement" : "disagreement", status: "pending", model: state.explanationSettings.model, message: "Waiting for the local explanation model. Scores stay unchanged." });
    }
  }

  function renderExplanation(record, container) {
    container.replaceChildren();
    container.append(node("h4", "", "Local explanation review"));
    if (record.status === "ok") {
      if (record.mode === "disagreement") {
        const bestFit = record.verdict === "best_fit" ? record.best_fit_level : null;
        container.append(node("p", "explanation-verdict", bestFit ? `Advisory best fit: ${bestFit.label} · ${fmt(bestFit.points)} rubric points` : "Uncertain · no best-fit rubric level selected"));
        container.append(node("p", "field-help", "This is an explanation of the disagreement, not a replacement grade. Your instructor score stays independent."));
      } else {
        const verdict = { agree: "Agrees with the shared rating", disagree: "Disagrees with the shared rating", uncertain: "Uncertain about the shared rating" }[record.verdict] || "Review returned";
        container.append(node("p", "explanation-verdict", verdict));
      }
      container.append(node("p", "explanation-text", record.explanation || ""));
      if (record.evidence?.length) {
        const evidence = node("details", "explanation-evidence"); evidence.append(node("summary", "", "Quoted submission evidence"));
        for (const quote of record.evidence) evidence.append(node("blockquote", "", quote));
        container.append(evidence);
      }
    } else {
      container.append(node("p", "explanation-text", record.message || "The local explanation is unavailable."));
      if (record.status === "ineligible") container.append(node("p", "field-help", "Grade again to create a current comparison before requesting an explanation."));
      const context = state.explanationContext;
      if (record.status === "unavailable" && currentExplanation(context)) {
        const retry = node("button", "small-button retry-explanation", "Retry explanation"); retry.type = "button";
        retry.addEventListener("click", () => { if (record.status === "unavailable") void requestExplanation(record, context); });
        container.append(retry);
      }
    }
    container.append(node("p", "field-help", `${record.model || "Local Ollama model"} · Explanation only. Model and instructor scores are unchanged.`));
  }

  function currentExplanation(context) {
    return !!context && state.explanationContext === context && state.revision === context.revision && state.pageEpoch === context.pageEpoch && state.comparison === context.comparison;
  }

  async function requestExplanation(record, context) {
    if (!currentExplanation(context) || context.inFlight.has(record.criterion_id)) return;
    const { payload, comparison, settings } = context;
    const container = state.explanationElements.get(record.criterion_id);
    const receipt = comparison.comparison?.find((row) => row.id === record.criterion_id)?.comparison_receipt;
    if (!receipt) {
      Object.assign(record, { status: "ineligible", message: "This comparison cannot be used for an explanation. It may come from an older app version or no longer match the current configuration." });
      if (container) renderExplanation(record, container);
      return;
    }
    context.inFlight.add(record.criterion_id);
    record.status = "pending";
    record.message = `${record.mode === "disagreement" ? "The local explanation model is reviewing the disagreement and considering the existing rubric levels." : "The local explanation model is reviewing the shared below-full-credit rating."} This may take a while; you can continue your own review.`;
    if (container) renderExplanation(record, container);
    const decisions = comparison.results.map((entry, index) => { const row = entry.result.criteria.find((item) => item.id === record.criterion_id); const model = payload.models[index]; return { model: entry.result.model, provider: entry.result.provider, endpoint: model.provider === "systemone" ? model.endpoint : null, level_id: row.level_id, points: row.points, ok: true }; });
    try {
      const response = await api("/api/explain", jsonRequest({ rubric: payload.rubric, text: payload.text, assignment_details: payload.assignment_details, grading_guidance: payload.grading_guidance, criterion_id: record.criterion_id, decisions, comparison_receipt: receipt, settings }));
      if (!currentExplanation(context)) return;
      Object.assign(record, response, { criterion_id: record.criterion_id });
      if (record.status === "ok") delete record.message;
    } catch (error) {
      if (!currentExplanation(context)) return;
      Object.assign(record, { status: "unavailable", message: error.message });
    } finally { context.inFlight.delete(record.criterion_id); }
    if (container) renderExplanation(record, container);
  }

  async function explainComparison(payload, comparison, revision, pageEpoch) {
    const context = { payload, comparison, revision, pageEpoch, settings: clone(state.explanationSettings), inFlight: new Set() };
    state.explanationContext = context;
    for (const record of [...state.explanations]) {
      if (!currentExplanation(context)) return;
      await requestExplanation(record, context);
    }
  }

  function renderResults() {
    const content = $("results-content");
    content.replaceChildren();
    const entries = state.comparison.results;
    const totals = node("div", "comparison-totals");
    for (const entry of entries) {
      const card = node("article", `comparison-total${entry.ok ? "" : " model-failed"}`);
      card.append(node("p", "model-label", entry.label));
      if (entry.ok) {
        const incomplete = entry.result.total == null;
        const score = node("div", "comparison-score", incomplete ? "—" : fmt(entry.result.total)); score.append(node("span", "", ` / ${fmt(entry.result.max_points)}`)); card.append(score);
        if (incomplete) {
          const count = entry.result.review_required_criteria ?? entry.result.criteria.filter((criterion) => criterion.points == null).length;
          card.append(node("p", "model-abstention-summary", `Review required · ${count} of ${entry.result.criteria.length} criteria have no suggested score. No total assigned.`));
        } else {
          const reviewCount = entry.result.criteria.filter((criterion) => criterion.needs_review).length;
          card.append(node("p", "", `${entry.result.max_points ? fmt(entry.result.total / entry.result.max_points * 100) : "0"}% · ${reviewCount ? "Provisional model suggestion" : "Model suggestion"}`));
          if (reviewCount) card.append(node("p", "model-review-summary", `${reviewCount} of ${entry.result.criteria.length} criteria are uncertain and recommended for review.`));
        }
        card.append(node("p", "model-identifier", entry.result.model || entry.result.provider || "Local model"));
        for (const warning of entry.result.warnings || []) card.append(node("p", "model-warning", warning));
      } else {
        card.append(node("h3", "", "Model unavailable"));
        card.append(node("p", "", entry.error || "No result returned. Check local model settings."));
      }
      totals.append(card);
    }
    content.append(totals);
    for (const warning of state.comparison.warnings || []) content.append(node("p", "notice", warning));
    $("results-empty").hidden = true; content.hidden = false; $("result-actions").hidden = !state.result;
    if (!state.result) { content.append(node("p", "notice", "No local model returned a grade. Check the model configurations and try again. No score has been assigned.")); return; }
    const result = state.result;
    const context = node("details", "grading-guidance-context");
    context.append(node("summary", "", "Instructor grading guidance used"), node("p", "criterion-guidance", (result.grading_guidance ?? state.gradingGuidance) || "No additional instructor grading guidance was provided."));
    content.append(context);
    if (state.explanationSettings?.enabled) content.append(node("p", "comparison-note", "Local Ollama reviews disagreements or unanimous below-full-credit ratings when at least two selected models all return assigned levels. A disagreement review may identify an advisory best-fit rubric level; no model or instructor scores change."));
    const overview = node("div", "result-overview");
    const score = node("div", "result-score");
    const total = node("strong", "", "—"); total.id = "final-total"; score.append(total, node("span", "", `/ ${fmt(result.max_points)}`));
    const summary = node("div", "result-summary");
    const heading = node("h3", "", ""); heading.id = "final-percent";
    const explanation = node("p", "", ""); explanation.id = "original-total"; summary.append(heading, explanation);
    overview.append(score, summary); content.append(overview);
    const comparable = (state.comparison.comparison || []).filter((item) => typeof item.disagrees === "boolean");
    const differences = comparable.filter((item) => item.disagrees).length;
    const comparisonText = comparable.length ? `${differences} of ${comparable.length} criteria with at least two available ratings differ. Withheld scores and failed models are excluded; agreement does not establish fairness or accuracy.` : "No criterion has a suggested score from at least two models. Review the top preferences and probabilities below; uncertainty is not a zero grade.";
    const availableModels = entries.filter((entry) => entry.ok).length;
    content.append(node("p", "comparison-note", `${availableModels < entries.length ? `${availableModels} of ${entries.length} models returned predictions. This comparison is incomplete. ` : ""}${comparisonText}`));
    const list = node("div", "result-list comparison-list");
    for (const criterion of result.criteria) {
      const card = node("article", "result-card");
      const head = node("div", "result-card-head"); head.append(node("h3", "", criterion.name));
      const comparison = (state.comparison.comparison || []).find((item) => item.id === criterion.id);
      const assignedCount = entries.filter((entry) => entry.ok && entry.result.criteria.some((row) => row.id === criterion.id && row.points != null)).length;
      const modelScope = assignedCount === entries.length ? `All ${assignedCount} models` : `${assignedCount} available models`;
      if (assignedCount >= 2 && comparison?.disagrees) head.append(node("span", "review-badge", `Disagreement among ${assignedCount} available models`));
      else if (assignedCount >= 2 && comparison?.disagrees === false) head.append(node("span", "agreement-badge", `${modelScope} agree`));
      else if (entries.some((entry) => entry.ok && entry.result.criteria.some((row) => row.id === criterion.id && row.points == null))) head.append(node("span", "review-badge", "Score withheld"));
      card.append(head);
      const guidance = (result.rubric || state.rubric).criteria.find((item) => item.id === criterion.id)?.description;
      if (guidance) card.append(node("p", "criterion-guidance", guidance));
      const pair = node("div", "decision-pair"); for (const entry of entries) pair.append(modelDecision(entry, criterion.id)); card.append(pair);
      const controls = node("div", "review-controls");
      const overrideField = node("div"); const overrideId = uid("override"); const overrideLabel = node("label", "", "Instructor score"); overrideLabel.htmlFor = overrideId;
      const override = node("input"); override.type = "number"; override.id = overrideId; override.min = "0"; override.max = String(criterion.max_points); override.step = "any"; override.placeholder = "Enter score"; override.setAttribute("aria-label", `Instructor score for ${criterion.name}, 0 to ${fmt(criterion.max_points)} points`); override.setAttribute("autocomplete", "off");
      const noteField = node("div"); const noteId = uid("note"); const noteLabel = node("label", "", "Your review note"); noteLabel.htmlFor = noteId;
      const note = node("textarea"); note.id = noteId; note.rows = 1; note.placeholder = "Explain your decision or note a concern…"; note.setAttribute("autocomplete", "off");
      const overrideMessage = node("p", "override-label", "Enter your own score after reviewing the model suggestions.");
      state.overrides[criterion.id] = { points: null, note: "" };
      override.addEventListener("input", () => {
        const value = override.value === "" ? null : Number(override.value);
        if (value !== null && (!Number.isFinite(value) || value < 0 || value > criterion.max_points)) { override.setCustomValidity(`Enter a score between 0 and ${criterion.max_points}.`); overrideMessage.textContent = `Use a score from 0 to ${fmt(criterion.max_points)}.`; state.overrides[criterion.id].points = null; }
        else { override.setCustomValidity(""); state.overrides[criterion.id].points = value; overrideMessage.textContent = value === null ? "Enter your own score after reviewing the model suggestions." : `Confirmed by you: ${fmt(value)} / ${fmt(criterion.max_points)}. All model suggestions remain unchanged.`; }
        refreshTotal();
      });
      note.addEventListener("input", () => { state.overrides[criterion.id].note = note.value; });
      overrideField.append(overrideLabel, override); noteField.append(noteLabel, note); controls.append(overrideField, noteField);
      card.append(controls, overrideMessage);
      const explanation = state.explanations.find((record) => record.criterion_id === criterion.id);
      if (explanation) {
        const explanationBlock = node("section", "explanation-block"); explanationBlock.setAttribute("aria-label", `Local explanation for ${criterion.name}`); explanationBlock.setAttribute("aria-live", "polite");
        state.explanationElements.set(criterion.id, explanationBlock); renderExplanation(explanation, explanationBlock); card.append(explanationBlock);
      }
      list.append(card);
    }
    content.append(list, node("p", "model-footnote", "All models evaluate the same rubric and submission independently. Provisional points use the most likely rubric level, including when uncertain; optional strict review withholds uncertain scores. Top preferences and full probabilities remain visible. Scores are never averaged. Probabilities are not verified accuracy, and comparing models does not eliminate bias."));
    refreshTotal();
  }

  function exportResult() {
    if (!state.result) throw new Error("Compare a submission before exporting a result.");
    for (const input of $("results-content").querySelectorAll("input")) { if (!input.checkValidity()) { input.reportValidity(); throw new Error("Correct the review score before exporting."); } }
    const total = state.result.criteria.reduce((sum, criterion) => sum + (adjustedPoints(criterion) ?? 0), 0);
    const confirmed = state.result.criteria.every((criterion) => state.overrides[criterion.id]?.points !== null && state.overrides[criterion.id]?.points !== undefined);
    return { ...clone(state.comparison), explanations: clone(state.explanations), submission_name: state.result.submission_name, assignment_details: state.result.assignment_details ?? state.assignmentDetails, grading_guidance: state.result.grading_guidance ?? state.gradingGuidance, rubric: clone(state.result.rubric || state.rubric), review: { status: confirmed ? "instructor_confirmed" : "incomplete", total: confirmed ? total : null, partial_points: total, max_points: state.result.max_points, percentage: confirmed && state.result.max_points ? total / state.result.max_points * 100 : null, criteria: state.result.criteria.map((criterion) => ({ id: criterion.id, name: criterion.name, instructor_points: state.overrides[criterion.id]?.points ?? null, final_points: adjustedPoints(criterion), confirmed: state.overrides[criterion.id]?.points !== null && state.overrides[criterion.id]?.points !== undefined, note: state.overrides[criterion.id]?.note || "" })) } };
  }

  function csvCell(value) {
    let text = String(value ?? "");
    // Prevent spreadsheet formula execution in user-controlled titles and notes.
    if (/^[\s]*[=+\-@\t\r]/.test(text)) text = `'${text}`;
    return `"${text.replace(/"/g, '""')}"`;
  }

  function resultCsvRows(result) {
    const modelHeaders = (label) => [label, `${label} status`, `${label} score`, `${label} suggested level`, `${label} top preference`, `${label} top preference rubric points`, `${label} top preference probability`, `${label} review reason`, `${label} probabilities JSON`];
    const modelCells = (entry, criterionId) => {
      if (!entry.ok) return [entry.label, "unavailable", "", "", "", "", "", entry.error || "", ""];
      const criterion = entry.result.criteria.find((row) => row.id === criterionId);
      if (!criterion) return [entry.label, "unavailable", "", "", "", "", "", "No criterion result returned.", ""];
      const top = criterion.top_choice || { label: criterion.level_label, points: criterion.points, probability: criterion.confidence };
      return [entry.label, criterion.status || (criterion.points == null ? "review_required" : "suggested"), criterion.points ?? "", criterion.level_label || "", top.label || "", top.points ?? "", top.probability ?? "", criterion.review_reason || "", JSON.stringify(criterion.probabilities || [])];
    };
    const totalCells = (entry) => [entry.label, entry.ok ? entry.result.status || (entry.result.total == null ? "review_required" : "suggested") : "unavailable", entry.ok ? entry.result.total ?? "" : "", "", "", "", "", entry.ok ? "" : entry.error || "", ""];
    const explanationHeaders = ["Explanation status", "Explanation mode", "Explanation verdict", "Explanation", "Explanation evidence JSON", "Explanation model", "Advisory best-fit level ID", "Advisory best-fit label", "Advisory best-fit rubric points"];
    const explanationCells = (id) => { const record = result.explanations?.find((item) => item.criterion_id === id); return record ? [record.status, record.mode || "agreement", record.verdict || "", record.explanation || record.message || "", JSON.stringify(record.evidence || []), record.model || "", record.best_fit_level?.level_id || "", record.best_fit_level?.label || "", record.best_fit_level?.points ?? ""] : explanationHeaders.map(() => ""); };
    const rows = [["Submission", "Rubric", "Assignment details", "Instructor grading guidance", "Criterion", ...result.results.flatMap((entry, index) => modelHeaders(`Model ${modelLetter(index)}`)), "Maximum points", "Instructor score", "Review score", "Review status", "Review note", ...explanationHeaders]];
    for (const review of result.review.criteria) {
      const criterion = result.rubric.criteria.find((row) => row.id === review.id);
      rows.push([result.submission_name, result.rubric.title, result.assignment_details, result.grading_guidance ?? "", review.name, ...result.results.flatMap((entry) => modelCells(entry, review.id)), criterion ? maxFor(criterion) : "", review.instructor_points ?? "", review.final_points ?? "", review.confirmed ? "instructor_confirmed" : "incomplete", review.note, ...explanationCells(review.id)]);
    }
    rows.push([result.submission_name, result.rubric.title, result.assignment_details, result.grading_guidance ?? "", "TOTAL", ...result.results.flatMap(totalCells), result.review.max_points, result.review.status === "instructor_confirmed" ? result.review.total : "", result.review.total ?? "", result.review.status, "", ...explanationCells(null)]);
    return rows;
  }

  async function grade() {
    if (state.busy || state.vision.busy) return;
    announce("");
    try { validate(); } catch (error) { announce(error.message, true); return; }
    const revision = state.revision;
    const pageEpoch = state.pageEpoch;
    const models = currentModels();
    const payload = { rubric: clone(state.rubric), assignment_details: state.assignmentDetails, grading_guidance: state.gradingGuidance, models: clone(models), text: $("submission-text").value, submission_name: $("submission-name").value.trim() || "Untitled submission" };
    state.grading = true; setBusy(true);
    $("grade-button-label").textContent = "Comparing locally…";
    $("grade-button-icon").textContent = "";
    $("grade-button-icon").classList.add("spinner");
    $("grading-status").classList.add("busy-text");
    const start = Date.now();
    const progress = () => { if (revision === state.revision) status("grading-status", `Running ${models.length} local models against ${payload.rubric.criteria.length} ${payload.rubric.criteria.length === 1 ? "criterion" : "criteria"}… ${Math.floor((Date.now() - start) / 1000)}s${Date.now() - start > 15000 ? " · First model load can take longer." : ""}`); };
    progress(); const timer = setInterval(progress, 1000);
    try {
      const result = await api("/api/compare", jsonRequest(payload));
      if (revision !== state.revision) { if (pageEpoch === state.pageEpoch) status("grading-status", "Inputs changed during grading. Grade again to evaluate the current version."); return; }
      if (!Array.isArray(result.results) || result.results.length !== models.length) throw new Error("The server returned an incomplete model list. Open the updated application and grade again.");
      state.comparison = result;
      const firstSuccessful = result.results.find((entry) => entry.ok);
      state.result = firstSuccessful?.result || null;
      state.overrides = {};
      prepareExplanations(result);
      renderResults();
      if (state.explanations.length) void explainComparison(payload, result, revision, pageEpoch);
      const completed = result.results.filter((entry) => entry.ok).length;
      status("grading-status", `${completed} / ${models.length} models returned predictions in ${Math.max(1, Math.round((Date.now() - start) / 1000))}s. ${completed === models.length ? "Compare the results below; uncertain scores require review." : "Review the predictions and local model errors below."}`, completed === 0);
      $("results-section").scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
    } catch (error) { if (revision === state.revision) status("grading-status", error.message, true); }
    finally { clearInterval(timer); state.grading = false; $("grading-status").classList.remove("busy-text"); updateModelSummary(); $("grade-button-icon").classList.remove("spinner"); $("grade-button-icon").textContent = "→"; setBusy(false); }
  }

  function clearSubmission() {
    clearVision();
    $("submission-text").value = "";
    $("submission-name").value = "";
    $("submission-file").value = "";
    showWarnings("extraction-warnings", []);
    invalidate(); updateTextMeta(); status("grading-status", "Ready for the next student. Previous text, filename, and grade cleared; your rubric, assignment details, and instructor guidance are kept.");
    announce("");
    $("submission-text").focus();
  }

  function bindEvents() {
    $("vision-file").addEventListener("change", () => {
      const files = $("vision-file").files;
      const file = files[0];
      const count = files.length;
      $("vision-file").value = "";
      if (count > 1) { clearVision(); status("vision-status", "Choose one image or document at a time.", true); return; }
      if (file) return prepareVision(file);
    });
    $("vision-show-page").addEventListener("click", () => {
      if (state.vision.busy || !state.vision.file) return;
      const index = Number($("vision-index").value);
      if (!Number.isInteger(index) || index < 1 || index > state.vision.prepared?.count) { status("vision-status", "Choose a page or image number within this document.", true); return; }
      return prepareVision(state.vision.file, index, state.vision.detected, undefined, state.vision.nativeImported);
    });
    $("vision-detected-select").addEventListener("change", () => {
      if (state.vision.busy || !state.vision.file) return;
      const item = state.vision.detected.find((entry) => visionItemKey(entry) === $("vision-detected-select").value);
      if (item) return prepareVision(state.vision.file, item.index, state.vision.detected, item.image_index, state.vision.nativeImported);
    });
    $("vision-image-index").addEventListener("change", () => {
      if (state.vision.busy || !state.vision.file || !state.vision.prepared) return;
      const imageIndex = Number($("vision-image-index").value);
      if (Number.isInteger(imageIndex) && imageIndex >= 0 && imageIndex <= state.vision.prepared.image_count) return prepareVision(state.vision.file, state.vision.prepared.index, state.vision.detected, imageIndex, state.vision.nativeImported);
    });
    $("vision-analyze").addEventListener("click", analyzeVision);
    $("vision-append").addEventListener("click", appendVision);
    $("vision-clear").addEventListener("click", clearVision);
    $("vision-reviewed").addEventListener("change", renderVisionControls);
    for (const id of ["vision-transcription", "vision-description"]) $(id).addEventListener("input", () => { $("vision-reviewed").checked = false; state.vision.added = false; renderVisionControls(); });
    $("vision-model").addEventListener("input", () => {
      state.visionSettings.model = $("vision-model").value;
      clearVisionAnalysis();
      status("workspace-status", "Vision model changed in this page. Save the workspace explicitly to reuse it.");
    });
    $("vision-detailed-crops").addEventListener("change", () => {
      state.visionSettings.detailed_crops = $("vision-detailed-crops").checked;
      clearVisionAnalysis();
      status("workspace-status", "Detailed image reading changed. Save the workspace explicitly to reuse this setting.");
    });
    $("vision-auto-detect").addEventListener("change", () => {
      state.visionSettings.auto_detect = $("vision-auto-detect").checked;
      status("workspace-status", "Automatic image detection updated for document imports. Save the workspace explicitly to reuse this setting.");
    });
    $("explanation-enabled").addEventListener("change", () => { state.explanationSettings.enabled = $("explanation-enabled").checked; invalidate(); });
    $("explanation-model").addEventListener("input", () => { state.explanationSettings.model = $("explanation-model").value; invalidate(); });
    $("assignment-details").addEventListener("input", (event) => { state.assignmentDetails = event.target.value; invalidate(); });
    $("grading-guidance").addEventListener("input", (event) => { state.gradingGuidance = event.target.value; invalidate(); });
    $("rubric-title").addEventListener("input", (event) => { state.rubric.title = event.target.value; invalidate(); });
    $("rubric-instructions").addEventListener("input", (event) => { state.rubric.instructions = event.target.value; invalidate(); });
    $("submission-text").addEventListener("input", () => { invalidate(); updateTextMeta(); });
    $("submission-name").addEventListener("input", invalidate);
    $("add-criterion").addEventListener("click", () => { const id = uid("criterion"); state.rubric.criteria.push({ id, name: "New criterion", description: "", levels: [{ id: uid("level"), label: "Meets expectations", points: 5, description: "" }, { id: uid("level"), label: "Needs improvement", points: 0, description: "" }] }); state.expandedCriteria.add(id); invalidate(); renderRubric(); const input = $("criteria-editor").lastElementChild.querySelector(".criterion-name"); input.focus(); input.select(); });
    $("model-search").addEventListener("input", renderPicker);
    $("add-custom-model").addEventListener("click", () => addModel());
    $("save-workspace").addEventListener("click", () => workspaceAction("save"));
    $("load-workspace").addEventListener("click", () => workspaceAction("load"));
    $("import-workspace").addEventListener("click", () => $("workspace-file").click());
    $("workspace-file").addEventListener("change", () => { const file = $("workspace-file").files[0]; $("workspace-file").value = ""; if (file) void workspaceAction("import", file); });
    $("export-workspace").addEventListener("click", () => { try { download("margin-model-workspace.json", JSON.stringify(workspaceSnapshot(), null, 2), "application/json"); status("workspace-status", "Exported model, explanation, and vision settings only."); } catch (error) { status("workspace-status", error.message, true); } });
    $("model-settings").addEventListener("toggle", () => { if ($("model-settings").open) startRuntimePolling(); else stopRuntimePolling(); });
    $("refresh-runtimes").addEventListener("click", startRuntimePolling);
    $("probe-model").addEventListener("click", async () => {
      if (state.busy) return;
      try { validateModelSettings(); } catch (error) { status("model-status", error.message, true); return; }
      const revision = state.revision;
      const models = currentModels();
      setBusy(true); status("model-status", "Checking each local model…");
      try {
        const results = [];
        for (const [index, settings] of models.entries()) {
          if (revision !== state.revision) return;
          status("model-status", `Checking ${settings.display_name || settings.model || `Model ${modelLetter(index)}`}…`);
          try { results.push({ status: "fulfilled", value: await api("/api/probe", jsonRequest({ settings: clone(settings) }), 180000) }); }
          catch (reason) { results.push({ status: "rejected", reason }); }
        }
        if (revision === state.revision) status("model-status", results.map((entry, index) => `${models[index].display_name || models[index].model || `Model ${modelLetter(index)}`}: ${entry.status === "fulfilled" ? entry.value.message || (entry.value.ok ? "Ready." : "Not ready.") : entry.reason.message}`).join(" · "), results.some((entry) => entry.status === "rejected" || !entry.value?.ok));
      }
      catch (error) { if (revision === state.revision) status("model-status", error.message, true); } finally { setBusy(false); }
    });
    $("import-rubric").addEventListener("click", () => $("rubric-file").click());
    $("rubric-file").addEventListener("change", async () => {
      const file = $("rubric-file").files[0];
      $("rubric-file").value = "";
      if (!file) return;
      if (file.size > 15 * 1024 * 1024) { announce("This rubric exceeds the 15 MB import limit.", true); return; }
      if (/\.json$/i.test(file.name)) {
        const revision = state.revision;
        setBusy(true);
        try {
          const imported = JSON.parse(await file.text());
          if (revision !== state.revision) return;
          state.rubric = normalizeRubric(imported.rubric || imported); invalidate(); renderRubric(); announce(`Imported rubric from ${file.name}.`);
          status("grading-status", "");
        }
        catch (error) { if (revision === state.revision) announce(error instanceof SyntaxError ? "This file is not valid JSON. Choose a rubric JSON file or import a text document." : error.message, true); }
        finally { setBusy(false); }
      } else await extractFile(file, "rubric");
    });
    $("export-rubric").addEventListener("click", () => { if (state.rubric) download(`${safeFilename(state.rubric.title)}-rubric.json`, JSON.stringify(state.rubric, null, 2), "application/json"); });
    $("submission-file").addEventListener("change", async () => { const files = $("submission-file").files; const file = files[0], count = files.length; $("submission-file").value = ""; if (count > 1) { announce("Choose one submission file at a time.", true); return; } await extractFile(file, "submission"); });
    const drop = $("submission-drop");
    for (const type of ["dragenter", "dragover"]) drop.addEventListener(type, (event) => { event.preventDefault(); if (!state.busy) drop.classList.add("dragover"); });
    for (const type of ["dragleave", "drop"]) drop.addEventListener(type, (event) => { event.preventDefault(); drop.classList.remove("dragover"); });
    drop.addEventListener("drop", (event) => { if (event.dataTransfer.files.length > 1) { announce("Choose one submission file at a time.", true); return; } extractFile(event.dataTransfer.files[0], "submission"); });
    $("clear-submission").addEventListener("click", clearSubmission);
    $("forget-workspace").addEventListener("click", () => {
      state.rubric = clone(state.defaults.rubric); state.expandedCriteria.clear(); state.models = clone(state.defaults.models);
      state.explanationSettings = clone(state.defaults.explanationSettings);
      state.visionSettings = clone(state.defaults.visionSettings);
      state.assignmentDetails = ""; $("assignment-details").value = "";
      state.gradingGuidance = ""; $("grading-guidance").value = "";
      clearSubmission(); renderRubric(); renderSettings();
      $("rubric-file").value = ""; status("model-status", ""); announce("Workspace cleared. Your rubric, assignment details, and instructor guidance are blank, and default model settings are restored in this page. Any explicitly saved configuration is unchanged.");
    });
    $("grade-button").addEventListener("click", grade);
    $("export-result-json").addEventListener("click", () => { try { const result = exportResult(); download(`${safeFilename(result.submission_name)}-grade.json`, JSON.stringify(result, null, 2), "application/json"); } catch (error) { announce(error.message, true); } });
    $("export-result-csv").addEventListener("click", () => {
      try {
        const result = exportResult();
        const rows = resultCsvRows(result);
        download(`${safeFilename(result.submission_name)}-grade.csv`, "\uFEFF" + rows.map((row) => row.map(csvCell).join(",")).join("\r\n"), "text/csv;charset=utf-8");
      } catch (error) { announce(error.message, true); }
    });
    document.querySelector('a[href="#model-settings"]').addEventListener("click", () => { $("model-settings").open = true; });
    // Do not let the browser restore potentially sensitive submission text from a prior history entry.
    window.addEventListener("pagehide", () => {
      state.revision += 1;
      state.pageEpoch += 1;
      clearVision();
      stopRuntimePolling();
      for (const controller of activeRequests) controller.abort();
      $("submission-text").value = ""; $("submission-name").value = ""; $("submission-file").value = "";
      state.result = null; state.comparison = null; state.overrides = {}; state.explanations = []; state.explanationContext = null; state.explanationElements.clear();
      $("results-content").replaceChildren(); $("results-content").hidden = true; $("result-actions").hidden = true; $("results-empty").hidden = false;
      showWarnings("extraction-warnings", []); announce(""); status("grading-status", ""); status("model-status", ""); status("workspace-status", ""); status("runtime-status", "");
    });
    window.addEventListener("pageshow", (event) => { if (event.persisted) { $("results-empty").hidden = false; $("results-content").hidden = true; $("result-actions").hidden = true; updateTextMeta(); status("grading-status", ""); } });
  }

  async function init() {
    setBusy(true);
    // Disable browser autofill for text that can contain student data.
    for (const input of document.querySelectorAll("input,textarea")) input.setAttribute("autocomplete", "off");
    $("submission-text").value = ""; $("submission-name").value = ""; $("assignment-details").value = ""; $("grading-guidance").value = "";
    clearVision();
    try {
      const [defaultsResponse, catalogResponse, workspaceResponse] = await Promise.allSettled([api("/api/defaults", {}, 20000), api("/api/model-catalog", {}, 20000), api("/api/workspace", {}, 20000)]);
      if (defaultsResponse.status !== "fulfilled") throw defaultsResponse.reason;
      if (catalogResponse.status !== "fulfilled") throw catalogResponse.reason;
      const defaults = defaultsResponse.value, catalog = catalogResponse.value;
      if (!Array.isArray(defaults.comparison_models) || !defaults.comparison_models.length || defaults.comparison_models.length > 8) throw new Error("Local model defaults are unavailable. Restart the updated application, then open a fresh page.");
      if (!Array.isArray(catalog.models) || catalog.models.some((item) => !["laya", "clef", "startlux", "d3"].includes(item.id) || !item.settings)) throw new Error("The local model catalog is unavailable.");
      state.catalog = catalog.models.map((item) => ({ ...item, settings: modelSettings(item.settings) }));
      const presetIds = ["laya", "clef", "startlux", "d3"];
      const defaultWorkspace = normalizeWorkspace({ schema_version: 1, models: defaults.comparison_models.map((settings, index) => ({ id: `default-${index + 1}`, enabled: true, preset_id: presetIds[index] || null, settings: modelSettings(settings) })), explanation_settings: defaults.explanation_settings || { enabled: true, model: "qwen38-27b:q4_k_m" }, vision_settings: defaults.vision_settings || { model: DEFAULT_VISION_MODEL } });
      state.defaults = { rubric: normalizeRubric(defaults.rubric), models: clone(defaultWorkspace.models), explanationSettings: clone(defaultWorkspace.explanation_settings), visionSettings: clone(defaultWorkspace.vision_settings), gradingGuidance: typeof defaults.grading_guidance === "string" ? defaults.grading_guidance : "" };
      let workspace = defaultWorkspace;
      try {
        if (workspaceResponse.status !== "fulfilled") throw new Error("Unavailable");
        workspace = normalizeWorkspace(workspaceResponse.value.workspace);
        if (workspaceResponse.value.saved) status("workspace-status", "Your explicitly saved model configuration is loaded. Student work is never restored.");
      } catch { status("workspace-status", "Saved configuration could not be loaded. Default models are active in this page; the saved file has not been changed. Import a valid configuration or save explicitly to recover.", true); }
      state.models = workspace.models; state.explanationSettings = workspace.explanation_settings;
      state.visionSettings = workspace.vision_settings;
      state.rubric = clone(state.defaults.rubric);
      state.gradingGuidance = state.defaults.gradingGuidance; $("grading-guidance").value = state.gradingGuidance;
      renderRubric(); renderSettings(); bindEvents(); state.initialized = true; setBusy(false);
    } catch (error) { announce(`${error.message} Reload the page after starting the local application.`, true); $("provider-summary").textContent = "Server unavailable"; }
  }

  init();
})();
