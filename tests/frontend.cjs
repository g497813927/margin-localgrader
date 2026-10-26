const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { webcrypto } = require('node:crypto');
const original = fs.readFileSync(require('node:path').join(__dirname, '../grader/static/app.js'), 'utf8').replace(/\r\n/g, '\n');
function setup() {
  const elements = new Map(), windowEvents = new Map();
  class Element {
    constructor(tagName = 'div') { this.tagName = tagName; this.value = ''; this._text = ''; this.hidden = false; this.children = []; this.listeners = new Map(); this.classList = { add(){}, remove(){}, toggle(){} }; this.parentElement = { classList: this.classList }; }
    set id(value) { this._id = value; elements.set(value, this); } get id() { return this._id; }
    set type(value) { if (this.tagName === 'select') throw new TypeError('HTMLSelectElement.type is read-only'); this._type = value; } get type() { return this.tagName === 'select' ? 'select-one' : this._type; }
    set textContent(value) { this._text = String(value); this.children = []; } get textContent() { return this._text + this.children.map(item => item.textContent).join(''); }
    setAttribute(name,value) { (this.attributes ||= {})[name]=String(value); } removeAttribute(name) { delete (this.attributes ||= {})[name]; if (name === 'src') this.src = ''; } addEventListener(name, fn) { this.listeners.set(name, fn); }
    replaceChildren(...items) { this.children = items; } append(...items) { this.children.push(...items); }
    querySelectorAll(selector) { return this.children.flatMap(item => [(selector.startsWith('.') ? (item.className || '').split(' ').includes(selector.slice(1)) : item.tagName === selector) ? item : null, ...item.querySelectorAll(selector)]).filter(Boolean); }
    get lastElementChild() { return this.children.at(-1) || null; }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    focus() {} select() {} click() { return this.listeners.get('click')?.(); } remove() {} scrollIntoView() {} checkValidity() { return true; } setCustomValidity() {}
  }
  const get = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
  let pending; const requests = [];
  const context = vm.createContext({
    document: { getElementById: get, createElement: tag => new Element(tag), querySelector: () => new Element(), querySelectorAll: () => [], body: new Element() },
    window: { addEventListener: (name, fn) => windowEvents.set(name, fn), matchMedia: () => ({ matches: true }) },
    fetch: (url, options) => new Promise(resolve => { pending = { url, options, resolve }; requests.push(pending); }),
    FormData: class { constructor(){this.values=new Map();} append(key,value){this.values.set(key,value);} get(key){return this.values.get(key);} }, URL, Blob, AbortController, setTimeout, clearTimeout, setInterval, clearInterval, crypto: webcrypto
  });
  const code = original.replace('  init();\n})();', '  globalThis.__test = { state, activeRequests, bindEvents, grade, extractFile, prepareVision, analyzeVision, appendVision, clearVision, exportResult, resultCsvRows, refreshTotal, clearSubmission, normalizeRubric, validate, renderSettings, renderPicker, renderResults, init, modelSettings, currentModels, selectedEntries, addModel, moveModel, removeModel, workspaceSnapshot, normalizeWorkspace, applyWorkspace, workspaceAction, refreshRuntimes, runtimeAction, stopRuntimePolling };\n})();');
  vm.runInContext(code, context);
  const test = context.__test;
  test.state.rubric = { title: 'Synthetic rubric', instructions: '', criteria: [{ id: 'c', name: 'Reasoning', description: 'Explain the reasoning clearly.', levels: [{ id:'yes', label:'Demonstrated', description:'', points:9 }, {id:'no',label:'Not demonstrated',description:'',points:0}] }] };
  test.state.models = [configEntry(test,1,{provider:'laya',model:'local-test'}),configEntry(test,2,{provider:'systemone',endpoint:'http://127.0.0.1:1235/v1/systemone',model:'clef-test'})];
  test.state.explanationSettings={enabled:false,model:'qwen38-27b:q4_k_m'};
  test.state.initialized = true;
  const result = {total:9,max_points:9,criteria:[{id:'c',name:'Reasoning',points:9,max_points:9}],rubric:test.state.rubric,submission_name:'Synthetic student',assignment_details:'Synthetic assignment',grading_guidance:'Supportive guidance used for this grade'};
  return { test, get, windowEvents, result, requests, pending:()=>pending };
}
function configEntry(test,index,settings) { return {id:`m${index}`,enabled:true,preset_id:null,settings:test.modelSettings(settings)}; }
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const respond=(request,data,ok=true)=>request.resolve({ok,json:async()=>data});
(async () => {
  {
    const {test,get}=setup();
    test.renderSettings();
    assert.equal(get('provider-summary').textContent,'2 enabled · 2 configured');
    get('model-m2').value='custom-startlux'; get('model-m2').listeners.get('input')();
    assert.equal(test.state.models[1].settings.model,'custom-startlux');
    get('provider-m1').value='systemone';get('provider-m1').listeners.get('change')();
    assert.equal(test.state.models[0].settings.provider,'systemone');
    const html=fs.readFileSync(require('node:path').join(__dirname,'../grader/static/index.html'),'utf8');
    assert.doesNotMatch(html,/Model B · Clef|Clef local endpoint|Clef model identifier/);
    assert.match(original,/System One · local decision server/);
    assert.match(html,/Find a local decision model/);
    assert.doesNotMatch(html + original,/CS2|first-year|Try a sample|sample-essay|const SAMPLE|A university is considering software/,'The public UI contains no course policy or sample essay flow');
    assert.match(html,/optional · editable/);
    assert.doesNotMatch(html,/id="model-card-[abcd]"/);
    const guidanceDetails=html.match(/<details\b([^>]*)>\s*<summary>Instructor grading guidance[\s\S]*?<\/details>/);
    assert.ok(guidanceDetails,'Instructor guidance remains in a native expandable details section');
    assert.doesNotMatch(guidanceDetails[1],/\bopen(?:\s|=|$)/,'Guidance is collapsed by default');
    assert.match(guidanceDetails[0],/<textarea id="grading-guidance"/,'Collapsed content retains the active guidance field');
  }
  // The generic app starts blank, accepts a user-built rubric, and resets to blank again.
  {
    const {test,get,requests,pending}=setup();
    const blank={title:'',instructions:'',criteria:[]};
    const defaults={rubric:blank,grading_guidance:'',comparison_models:test.currentModels(),explanation_settings:test.state.explanationSettings};
    const workspace={schema_version:1,models:test.state.models,explanation_settings:test.state.explanationSettings};
    get('submission-text').value='Autofill must clear';get('grading-guidance').value='Old guidance';get('assignment-details').value='Old assignment';
    const initializing=test.init();
    respond(requests.find(request=>request.url==='/api/defaults'),defaults);
    respond(requests.find(request=>request.url==='/api/model-catalog'),{models:[]});
    respond(requests.find(request=>request.url==='/api/workspace'),{workspace,saved:false});await initializing;
    assert.equal(test.state.initialized,true);
    assert.deepEqual(JSON.parse(JSON.stringify(test.state.rubric)),blank);
    assert.deepEqual(JSON.parse(JSON.stringify(test.normalizeRubric({criteria:[]}))),blank,'Missing titles stay blank too');
    for(const id of ['rubric-title','rubric-instructions','assignment-details','grading-guidance','submission-text']) assert.equal(get(id).value,'',id+' starts blank');
    assert.equal(test.state.gradingGuidance,'');assert.equal(get('rubric-empty').hidden,false);assert.equal(get('criteria-editor').children.length,0);assert.equal(get('rubric-total').textContent,'0 points');
    get('submission-text').value='A synthetic response to a user-defined task.';
    const initialRequests=requests.length;await test.grade();assert.equal(requests.length,initialRequests,'An empty rubric sends no grading request');assert.match(get('global-status').textContent,/Give your rubric a title/);
    get('rubric-title').value='My custom rubric';get('rubric-title').listeners.get('input')({target:get('rubric-title')});
    await test.grade();assert.equal(requests.length,initialRequests);assert.match(get('global-status').textContent,/between 1 and 32 criteria/);
    get('add-criterion').listeners.get('click')();
    assert.equal(test.state.rubric.criteria.length,1);assert.equal(get('rubric-empty').hidden,true);
    const editor=get('criteria-editor'),name=editor.querySelector('.criterion-name'),description=editor.querySelector('.criterion-description');
    name.value='Clarity';name.listeners.get('input')();description.value='The response explains its reasoning in understandable terms.';description.listeners.get('input')();
    const points=editor.querySelector('.level-points');points.value='10';points.listeners.get('input')();
    assert.doesNotThrow(()=>test.validate(),'The Add a criterion workflow can create a valid rubric');
    assert.equal(get('rubric-total').textContent,'10 points');
    get('rubric-instructions').value='Apply the listed levels.';get('rubric-instructions').listeners.get('input')({target:get('rubric-instructions')});
    get('grading-guidance').value='Optional user instructions';get('grading-guidance').listeners.get('input')({target:get('grading-guidance')});
    const built=JSON.parse(JSON.stringify(test.state.rubric));
    const grading=test.grade(),payload=JSON.parse(pending().options.body);assert.deepEqual(payload.rubric,built);assert.equal(payload.grading_guidance,'Optional user instructions');
    const criterion=built.criteria[0];
    const result={total:10,max_points:10,rubric:built,submission_name:'Synthetic response',assignment_details:'',grading_guidance:payload.grading_guidance,criteria:[{id:criterion.id,name:criterion.name,points:10,max_points:10,level_id:criterion.levels[0].id,level_label:criterion.levels[0].label,confidence:0.9,probabilities:[]}]};
    respond(pending(),{results:payload.models.map(model=>({label:model.model,ok:true,result})),comparison:[]});await grading;
    assert.equal(test.exportResult().rubric.title,'My custom rubric');
    get('forget-workspace').listeners.get('click')();
    assert.deepEqual(JSON.parse(JSON.stringify(test.state.rubric)),blank);assert.equal(test.state.gradingGuidance,'');
    assert.equal(get('rubric-title').value,'');assert.equal(get('rubric-instructions').value,'');assert.equal(get('grading-guidance').value,'');assert.equal(get('rubric-empty').hidden,false);assert.equal(test.state.result,null);
    get('rubric-file').files=[{name:'my-rubric.json',size:1000,text:async()=>JSON.stringify(built)}];await get('rubric-file').listeners.get('change')();
    assert.deepEqual(JSON.parse(JSON.stringify(test.state.rubric)),built,'JSON import still restores user-authored criteria');assert.equal(get('rubric-empty').hidden,true);
    get('criteria-editor').querySelector('.criterion-head').querySelector('button').listeners.get('click')();
    assert.equal(test.state.rubric.criteria.length,0);assert.equal(get('rubric-empty').hidden,false,'Removing the final criterion restores the empty-rubric guidance');
    assert.equal(requests.length,initialRequests+1,'Reset, editing, and rubric import never save or grade automatically');
  }
  {
    const {test,get,result} = setup();
    test.state.result=result; test.state.comparison={results:[{ok:true,label:'A',result},{ok:false,label:'B',error:'offline'}],comparison:[]};
    test.state.overrides={c:{points:null,note:''}};
    const incomplete=test.exportResult();
    assert.equal(incomplete.review.total,null); assert.equal(incomplete.review.percentage,null);
    assert.equal(incomplete.review.criteria[0].final_points,null); assert.equal(incomplete.review.status,'incomplete');
    assert.equal(incomplete.assignment_details,'Synthetic assignment');
    assert.equal(incomplete.grading_guidance,'Supportive guidance used for this grade');
    const rows=test.resultCsvRows(incomplete), guidanceColumn=rows[0].indexOf('Instructor grading guidance');
    assert.ok(guidanceColumn >= 0);
    assert.ok(rows.slice(1).every(row=>row[guidanceColumn] === incomplete.grading_guidance));
    test.refreshTotal(); assert.equal(get('final-total').textContent,'—');
    test.state.overrides.c.points=0;
    const confirmed=test.exportResult(); assert.equal(confirmed.review.total,0); assert.equal(confirmed.review.percentage,0); assert.equal(confirmed.review.status,'instructor_confirmed');
    test.state.assignmentDetails='Keep assignment'; test.state.gradingGuidance='Keep course guidance'; get('grading-guidance').value='Keep course guidance'; get('submission-text').value='Synthetic student text';get('submission-name').value='Synthetic student';
    assert.equal(test.exportResult().grading_guidance,result.grading_guidance,'Export uses the evaluated guidance snapshot');
    test.clearSubmission();assert.equal(test.state.result,null);assert.equal(get('submission-text').value,'');assert.equal(test.state.assignmentDetails,'Keep assignment');
    assert.equal(test.state.gradingGuidance,'Keep course guidance');assert.equal(get('grading-guidance').value,'Keep course guidance');
  }
  {
    const {test,get,requests}=setup();
    const defaults={rubric:test.state.rubric,comparison_models:[...test.currentModels(),test.modelSettings({provider:'systemone',endpoint:'http://127.0.0.1:1236/v1/systemone',model:'startlux-test'}),test.modelSettings({provider:'systemone',endpoint:'http://127.0.0.1:1238/v1/systemone',model:'d3-test'})],grading_guidance:'Optional grading guidance from the server',explanation_settings:{enabled:true,model:'qwen38-27b:q4_k_m'}};
    const catalog={models:defaults.comparison_models.map((settings,index)=>({id:['laya','clef','startlux','d3'][index],name:settings.model,description:'Local preset',settings}))};
    const saved={schema_version:1,models:[configEntry(test,1,{provider:'laya',model:'saved-custom-path'})],explanation_settings:{enabled:true,model:'deepseek-r1:70b'}};
    const initializing=test.init();
    respond(requests.find(r=>r.url==='/api/defaults'),defaults);respond(requests.find(r=>r.url==='/api/model-catalog'),catalog);respond(requests.find(r=>r.url==='/api/workspace'),{workspace:saved,saved:true});
    await initializing;
    assert.equal(test.state.gradingGuidance,defaults.grading_guidance);
    assert.equal(test.state.models.length,1);assert.equal(test.state.models[0].settings.model,'saved-custom-path');
    assert.equal(test.state.explanationSettings.model,'deepseek-r1:70b','Explicitly saved custom model takes precedence');
    assert.equal(get('explanation-model').value,'deepseek-r1:70b');
    test.clearSubmission();assert.equal(test.state.explanationSettings.model,'deepseek-r1:70b');
    assert.equal(test.state.defaults.explanationSettings.model,'qwen38-27b:q4_k_m','Reset uses server defaults');
    get('forget-workspace').listeners.get('click')();
    assert.equal(test.state.models.length,4);assert.equal(test.state.models[2].settings.model,'startlux-test');assert.equal(test.state.models[3].settings.model,'d3-test');
    assert.equal(test.state.explanationSettings.model,'qwen38-27b:q4_k_m');
    assert.equal(test.state.gradingGuidance,'');assert.equal(test.state.assignmentDetails,'');assert.equal(test.state.result,null);
    assert.equal(requests.length,3,'Reset never overwrites the saved file');
    assert.throws(()=>test.exportResult(),/Compare a submission/);
  }

  {
    const {test,get,result,pending}=setup();test.bindEvents();
    get('submission-text').value='Synthetic test text';
    get('grading-guidance').value='Reward clear understanding; reduce only for major issues.';
    get('grading-guidance').listeners.get('input')({target:get('grading-guidance')});
    const grading=test.grade();
    assert.equal(JSON.parse(pending().options.body).grading_guidance,test.state.gradingGuidance);
    const revision=test.state.revision;
    get('grading-guidance').value='Updated instructor guidance';
    get('grading-guidance').listeners.get('input')({target:get('grading-guidance')});
    assert.equal(test.state.revision,revision+1);
    pending().resolve({ok:true,json:async()=>({results:[{ok:true,label:'A',result},{ok:true,label:'B',result}],comparison:[]})});
    await grading;
    assert.equal(test.state.result,null);assert.equal(test.state.comparison,null);
    assert.throws(()=>test.exportResult(),/Compare a submission/);
    test.state.result=result;test.state.comparison={results:[{ok:true,label:'A',result},{ok:true,label:'B',result}],comparison:[]};
    get('grading-guidance').value='';get('grading-guidance').listeners.get('input')({target:get('grading-guidance')});
    assert.equal(test.state.gradingGuidance,'');assert.equal(test.state.result,null);
    const baseline=test.grade();
    assert.equal(JSON.parse(pending().options.body).grading_guidance,'','An empty policy is sent explicitly for unchanged baseline behavior');
    pending().resolve({ok:true,json:async()=>({results:[{ok:false,label:'A',error:'Synthetic failure'},{ok:false,label:'B',error:'Synthetic failure'}],comparison:[]})});
    await baseline;
    test.state.gradingGuidance='x'.repeat(12001);
    assert.throws(()=>test.validate(),/12,000 characters/);
  }
  {
    const {test,get,windowEvents,result,pending}=setup();
    test.bindEvents();get('submission-text').value='Synthetic student text';get('submission-name').value='Synthetic student';
    const grading=test.grade();
    assert.equal(pending().url,'/api/compare');assert.equal(test.activeRequests.size,1);
    assert.equal(pending().options.headers['X-Margin-Scoring-Version'],'3');
    assert.ok(JSON.parse(pending().options.body).models.every(model => model.scoring_mode === 'suggest'),'Provisional suggestions are the default for both models');
    const previous=test.state.revision;
    windowEvents.get('pagehide')();
    assert.equal(test.state.revision,previous+1);assert.equal(pending().options.signal.aborted,true);
    pending().resolve({ok:true,json:async()=>({results:[{ok:true,label:'A',result},{ok:false,label:'B',error:'offline'}],comparison:[]})});
    await grading;
    assert.equal(test.state.result,null);assert.equal(test.state.comparison,null);assert.equal(get('submission-text').value,'');assert.equal(get('submission-name').value,'');assert.equal(get('grading-status').textContent,'');assert.equal(test.activeRequests.size,0);
  }
  {
    const {test,get,windowEvents,pending}=setup();test.bindEvents();
    const extraction=test.extractFile({name:'synthetic.txt',size:20},'submission');
    windowEvents.get('pagehide')();
    pending().resolve({ok:true,json:async()=>({text:'Must not return',filename:'synthetic.txt',warnings:[]})});
    await extraction;assert.equal(get('submission-text').value,'');assert.equal(get('submission-name').value,'');assert.equal(get('global-status').textContent,'');
  }
  {
    const {test,get}=setup();get('submission-text').value='Synthetic text';
    test.state.rubric.criteria[0].levels.pop();assert.throws(()=>test.validate(),/between 2 and 12/);
    assert.throws(()=>test.normalizeRubric(test.state.rubric),/between 2 and 12/);
    test.state.rubric.criteria[0].description='';assert.throws(()=>test.validate(),/Describe what/);
  }
  for (const scenario of [
    { name:'Summary', maximum:9, topLabel:'Missing', topProbability:0.3523, levels:[['above','Above competent',9,0.2601],['competent','Competent',6,0.2559],['needs','Needs improvement',3,0.1317],['missing','Missing',0,0.3523]] },
    { name:'Grammar', maximum:2, topLabel:'Unclear response', topProbability:0.5389, levels:[['above','Above competent',2,0.4611],['missing','Unclear response',0,0.5389]] },
    { name:'Grammar positive preference', maximum:2, topLabel:'Above competent', topProbability:0.5389, levels:[['above','Above competent',2,0.5389],['missing','Unclear response',0,0.4611]] }
  ]) for (const scoringMode of ['suggest','strict_review']) {
    const {test,get,pending}=setup();
    const withheld = scoringMode === 'strict_review';
    const topLevel = scenario.levels.find((level) => level[1] === scenario.topLabel);
    const zeroLevel = scenario.levels.find((level) => level[2] === 0);
    test.state.models.forEach(entry=>entry.settings.scoring_mode=scoringMode);
    test.state.rubric.criteria[0].name = scenario.name;
    test.state.rubric.criteria[0].levels = scenario.levels.map(([id,label,points]) => ({id,label,points,description:''}));
    const probabilities = scenario.levels.map(([id,label,points,probability]) => ({id,label,points,probability}));
    const uncertain = {id:'c',name:scenario.name,points:withheld ? null : topLevel[2],max_points:scenario.maximum,level_id:withheld ? null : topLevel[0],level_label:withheld ? null : scenario.topLabel,description:'',confidence:scenario.topProbability,needs_review:true,status:withheld ? 'review_required' : 'provisional',review_reason:withheld ? 'Top probability is below the review threshold. No score was assigned.' : 'Top probability is below the review threshold; review this provisional score.',top_choice:{id:topLevel[0],label:scenario.topLabel,points:topLevel[2],description:'',probability:scenario.topProbability},probabilities};
    const confidentZero = {...uncertain,points:0,level_id:zeroLevel[0],level_label:zeroLevel[1],confidence:0.9,needs_review:false,status:'suggested',review_reason:null,top_choice:{id:zeroLevel[0],label:zeroLevel[1],points:0,description:'',probability:0.9}};
    const a = {total:withheld ? null : topLevel[2],percentage:withheld ? null : topLevel[2] / scenario.maximum * 100,max_points:scenario.maximum,suggested_subtotal:withheld ? 0 : topLevel[2],scored_criteria:withheld ? 0 : 1,review_required_criteria:1,status:withheld ? 'review_required' : 'provisional',criteria:[uncertain],rubric:test.state.rubric,submission_name:'Synthetic student',assignment_details:'Synthetic assignment'};
    const b = {...a,total:0,percentage:0,scored_criteria:1,review_required_criteria:0,status:'suggested',criteria:[confidentZero]};
    get('submission-text').value='Synthetic assessment fixture';
    const grading=test.grade();
    assert.equal(pending().options.headers['X-Margin-Scoring-Version'],'3');
    assert.ok(JSON.parse(pending().options.body).models.every(model => model.scoring_mode === scoringMode));
    pending().resolve({ok:true,json:async()=>({results:[{label:'A',ok:true,result:a},{label:'B',ok:true,result:b}],comparison:[{id:'c',name:scenario.name,model_a_points:withheld ? null : topLevel[2],model_b_points:0,disagrees:withheld ? null : topLevel[2] !== 0}]})});
    await grading;
    const content=get('results-content');
    assert.match(content.textContent,/Instructor grading guidance used/);
    const blocks=content.querySelectorAll('.model-decision-block');
    assert.equal(blocks.length,2);
    assert.match(blocks[0].textContent,withheld ? /Review required/ : /Provisional suggestion · review recommended/);
    assert.match(blocks[0].textContent,new RegExp(`Top preference: ${scenario.topLabel}`));
    assert.equal(blocks[0].querySelectorAll('.criterion-score').length,withheld ? 0 : 1);
    if (!withheld) {
      assert.equal(blocks[0].querySelectorAll('.criterion-score')[0].textContent,`${topLevel[2]}/ ${scenario.maximum}`);
      assert.match(content.textContent,/Provisional model suggestion/);
    }
    assert.equal(blocks[1].querySelectorAll('.criterion-score')[0].textContent,`0/ ${scenario.maximum}`,'A confident zero remains a real zero');
    assert.equal(content.querySelectorAll('.comparison-score')[0].textContent,`${withheld ? '—' : topLevel[2]} / ${scenario.maximum}`);
    assert.equal(get('final-total').textContent,'—');
    if (withheld) {
      assert.match(content.textContent,/No criterion has a suggested score from at least two models/);
      assert.equal(content.querySelectorAll('.agreement-badge').length,0);
    }
    const input=content.querySelectorAll('input')[0];
    assert.equal(input.value,'');assert.equal(input.placeholder,'Enter score');
    const exported=test.exportResult();
    assert.equal(exported.results[0].result.criteria[0].points,withheld ? null : topLevel[2]);
    assert.equal(exported.results[0].result.total,withheld ? null : topLevel[2]);
    assert.equal(exported.results[0].result.criteria[0].top_choice.points,topLevel[2]);
    assert.equal(exported.results[0].result.criteria[0].top_choice.probability,scenario.topProbability);
    assert.equal(exported.results[0].result.criteria[0].probabilities.length,scenario.levels.length);
    assert.equal(exported.review.total,null);
    const rows=test.resultCsvRows(exported), headers=rows[0], row=rows[1], total=rows[2];
    assert.ok(rows.every(item=>item.length===headers.length));
    assert.equal(row[headers.indexOf('Model A score')],withheld ? '' : topLevel[2]);
    assert.equal(row[headers.indexOf('Model A status')],withheld ? 'review_required' : 'provisional');
    assert.equal(row[headers.indexOf('Model A top preference')],scenario.topLabel);
    assert.equal(row[headers.indexOf('Model A top preference rubric points')],topLevel[2]);
    assert.equal(row[headers.indexOf('Model A top preference probability')],scenario.topProbability);
    assert.equal(JSON.parse(row[headers.indexOf('Model A probabilities JSON')]).length,scenario.levels.length);
    assert.equal(total[headers.indexOf('Model A score')],withheld ? '' : topLevel[2]);
    assert.equal(row[headers.indexOf('Model B score')],0);
    assert.equal(row[headers.indexOf('Instructor score')],'');
    input.value='0';input.listeners.get('input')();
    assert.equal(get('final-total').textContent,'0');
    const confirmed=test.exportResult();assert.equal(confirmed.review.total,0);assert.equal(confirmed.review.status,'instructor_confirmed');
    assert.equal(confirmed.results[0].result.criteria[0].points,withheld ? null : topLevel[2],'Instructor review never rewrites a model score');
  }
  // Dynamic entries determine enabled count, order, per-model preferences, and every export column.
  {
    const {test,get,result,requests,pending}=setup();test.renderSettings();test.bindEvents();
    get('submission-text').value='Synthetic dynamic workspace';
    get('enabled-m2').checked=false;get('enabled-m2').listeners.get('change')();
    const single=test.grade();assert.equal(JSON.parse(pending().options.body).models.length,1);
    respond(pending(),{results:[{label:'Only Laya',ok:true,result}],comparison:[{id:'c',model_a_points:9,model_b_points:null,disagrees:null}]});await single;
    assert.equal(get('results-content').querySelectorAll('.comparison-total').length,1);
    assert.equal(test.exportResult().results.length,1);assert.equal(test.state.explanations.length,0);
    const oldResult=test.state.result;
    for(let i=3;i<=5;i++) test.addModel({id:`custom-${i}`,name:`Local ${i}`,settings:test.modelSettings({provider:'systemone',model:`local-${i}`,endpoint:`http://127.0.0.1:${1200+i}/v1/systemone`})});
    assert.ok(oldResult);assert.equal(test.state.result,null,'Adding a model invalidates previous scores');
    get('enabled-m2').checked=true;get('enabled-m2').listeners.get('change')();
    const fifthId=test.state.models[4].id;
    for(let i=0;i<4;i++) test.moveModel(fifthId,-1);
    assert.equal(test.state.models[0].id,fifthId);
    get(`display_name-${fifthId}`).value='<b>My local model</b>';get(`display_name-${fifthId}`).listeners.get('input')();
    get(`confidence_threshold-${fifthId}`).value='80';get(`confidence_threshold-${fifthId}`).listeners.get('input')();
    get(`max_input_chars-${fifthId}`).value='100';get(`max_input_chars-${fifthId}`).listeners.get('input')();
    get(`scoring_mode-${fifthId}`).value='strict_review';get(`scoring_mode-${fifthId}`).listeners.get('change')();
    const configured=test.workspaceSnapshot();assert.equal(configured.models.length,5);
    const multiple=test.grade(),payload=JSON.parse(pending().options.body);
    assert.deepEqual(payload.models.map(model=>model.model),['local-5','local-test','clef-test','local-3','local-4']);
    assert.equal(payload.models[0].display_name,'<b>My local model</b>');assert.equal(payload.models[0].confidence_threshold,0.8);assert.equal(payload.models[0].max_input_chars,100);assert.equal(payload.models[0].scoring_mode,'strict_review');
    assert.equal(payload.models[1].confidence_threshold,0.65);assert.equal(payload.models[1].scoring_mode,'suggest','Threshold/mode changes are per model');
    const entries=payload.models.map((settings,index)=>index===2?{label:settings.model,ok:false,error:'Synthetic offline'}:{label:settings.display_name||settings.model,ok:true,result:{...result,model:settings.model,total:index===4?null:9,criteria:[{...result.criteria[0],points:index===4?null:9,level_id:index===4?null:'yes',needs_review:index===4,level_label:index===4?null:'Demonstrated'}]}});
    respond(pending(),{results:entries,comparison:[{id:'c',disagrees:false,available_models:3}]});await multiple;
    assert.equal(get('results-content').querySelectorAll('.comparison-total').length,5);
    assert.equal(get('results-content').querySelectorAll('.model-decision-block').length,5);
    assert.match(get('results-content').textContent,/3 available models agree/);
    assert.equal(get('results-content').querySelectorAll('b').length,0,'Display names are text, never HTML');
    const exported=test.exportResult(),csv=test.resultCsvRows(exported);assert.equal(exported.results.length,5);assert.equal(csv[1][csv[0].indexOf('Model E score')],'');
    assert.equal(csv[1][csv[0].indexOf('Model C status')],'unavailable');assert.ok(csv.every(row=>row.length===csv[0].length));
    test.removeModel('m1');assert.equal(test.state.result,null);assert.equal(test.state.models.length,4);
    while(test.state.models.length>2)test.removeModel(test.state.models.at(-1).id);
    const two=test.grade();assert.equal(JSON.parse(pending().options.body).models.length,2);respond(pending(),{results:entries.slice(0,2),comparison:[]});await two;
    assert.equal(test.exportResult().results.length,2,'Historical two-model exports remain supported');
    while(test.state.models.length)test.removeModel(test.state.models[0].id);
    assert.equal(test.workspaceSnapshot().models.length,0,'An empty configuration can be saved');
    const count=requests.length;await test.grade();assert.equal(requests.length,count);assert.match(get('global-status').textContent,/Enable at least one/);
    for(let i=0;i<9;i++) test.addModel({id:`custom-${i}`,settings:test.modelSettings({provider:'laya',model:`path-${i}`})});
    assert.equal(test.state.models.length,8);assert.match(get('workspace-status').textContent,/Eight models/);
  }
  // Save/load/import only contain whitelisted settings, preserve student context, and reject late changes.
  {
    const {test,get,result,requests,pending}=setup();test.renderSettings();test.bindEvents();
    test.state.assignmentDetails='PRIVATE assignment';test.state.gradingGuidance='PRIVATE guidance';get('submission-text').value='PRIVATE student';get('submission-name').value='PRIVATE name';
    test.state.rubric.title='PRIVATE rubric';test.state.models[0].student_text='PRIVATE extra';test.state.models[0].settings.secret='PRIVATE extra setting';
    test.state.result=result;test.state.comparison={results:[{ok:true,label:'A',result}],comparison:[]};test.state.overrides={c:{points:7,note:'PRIVATE review'}};
    const save=test.workspaceAction('save'),body=JSON.parse(pending().options.body);
    assert.equal(pending().url,'/api/workspace');assert.equal(pending().options.method,'PUT');
    assert.deepEqual(Object.keys(body).sort(),['explanation_settings','models','schema_version','vision_settings']);assert.doesNotMatch(JSON.stringify(body),/PRIVATE/);
    respond(pending(),{workspace:body,saved:true});await save;
    assert.equal(test.state.result,result,'Saving configuration does not change grading state');assert.equal(test.state.overrides.c.points,7);
    const replacement={...body,models:[body.models[1]],explanation_settings:{enabled:true,model:'another-installed:4b'}};
    const load=test.workspaceAction('load');respond(pending(),{workspace:replacement,saved:true});await load;
    assert.equal(test.state.models.length,1);assert.equal(test.state.models[0].id,'m2');assert.equal(test.state.explanationSettings.model,'another-installed:4b');
    assert.equal(get('submission-text').value,'PRIVATE student');assert.equal(get('submission-name').value,'PRIVATE name');assert.equal(test.state.rubric.title,'PRIVATE rubric');assert.equal(test.state.assignmentDetails,'PRIVATE assignment');assert.equal(test.state.gradingGuidance,'PRIVATE guidance');assert.equal(test.state.result,null,'Changing configuration invalidates old scores');
    const staleLoad=test.workspaceAction('load');get('display_name-m2').value='Newer name';get('display_name-m2').listeners.get('input')();respond(pending(),{workspace:body,saved:true});await staleLoad;
    assert.equal(test.state.models.length,1);assert.equal(test.state.models[0].settings.display_name,'Newer name');assert.match(get('workspace-status').textContent,/discarded/);
    const count=requests.length;
    await test.workspaceAction('import',{size:100,text:async()=>JSON.stringify({...body,text:'PRIVATE student'})});assert.equal(requests.length,count,'Unknown student fields are rejected before sending');
    assert.match(get('workspace-status').textContent,/Rubrics, submissions, and scores/);
    const raw=JSON.stringify(replacement,null,2),importing=test.workspaceAction('import',{size:raw.length,text:async()=>raw});await tick();
    assert.equal(pending().url,'/api/workspace/validate');assert.equal(pending().options.body,raw,'Raw JSON retains duplicate keys for server rejection');respond(pending(),{workspace:replacement});await importing;
    assert.equal(requests.filter(request=>request.options.method==='PUT').length,1,'Import and edits never save automatically');
    assert.throws(()=>test.normalizeWorkspace({...replacement,models:[replacement.models[0],replacement.models[0]]}),/unique ID/);
    const endpoint=JSON.parse(JSON.stringify(replacement));endpoint.models[0].settings.endpoint='https://remote.example/v1/systemone';assert.throws(()=>test.normalizeWorkspace(endpoint),/loopback/);
    assert.doesNotMatch(original,/localStorage|sessionStorage|indexedDB/);
  }
  // Local catalog search and bundled-runtime controls never invoke arbitrary endpoints or commands.
  {
    const {test,get,pending,requests,windowEvents}=setup();
    test.state.catalog=[{id:'laya',name:'Laya',description:'Embedded',settings:test.currentModels()[0],service_id:null},{id:'clef',name:'Clef',description:'Decision server',settings:test.currentModels()[1],service_id:'clef'}];
    test.state.models[1].settings.display_name='My custom reviewer';test.renderSettings();test.bindEvents();
    get('model-search').value='custom reviewer';get('model-search').listeners.get('input')();assert.match(get('model-search-results').textContent,/My custom reviewer/);assert.doesNotMatch(get('model-search-results').textContent,/Embedded/);assert.equal(requests.length,0);
    test.state.services=[{id:'clef',name:'Clef',status:'external',owned:false,can_start:false,can_stop:true,message:'Already running'},{id:'d3',name:'D3',status:'stopped',owned:false,can_start:true,can_stop:false}];test.renderSettings();
    assert.equal(get('stop-runtime-clef').disabled,true);await test.runtimeAction('clef','stop');await test.runtimeAction('anything','start');assert.equal(requests.length,0);
    const starting=test.runtimeAction('d3','start');assert.equal(pending().url,'/api/runtimes/d3/start');assert.equal(pending().options.body,undefined);
    await test.runtimeAction('d3','start');assert.equal(requests.length,1,'Duplicate launch clicks are ignored');
    respond(pending(),{service:{id:'d3',name:'D3',status:'starting',owned:true,can_start:false,can_stop:true}});await starting;
    assert.equal(pending().url,'/api/runtimes');respond(pending(),{services:[{id:'d3',name:'D3',status:'running',owned:true,ready:true,can_start:false,can_stop:true}]});await tick();
    assert.equal(get('stop-runtime-d3').disabled,false);assert.match(get('runtime-list').textContent,/running · ready/);
    const checking=test.refreshRuntimes();windowEvents.get('pagehide')();assert.equal(pending().options.signal.aborted,true);respond(pending(),{services:[{id:'clef',name:'Late service',status:'running'}]});await checking;
    assert.equal(test.state.services[0].id,'d3','Stale runtime responses never update a restored page');test.stopRuntimePolling();
  }
  // Explanations require the entire selected set, from two through eight; one model never triggers one.
  for(const count of [1,2,5]) {
    const {test,get,result,pending,requests}=setup();
    test.state.models=Array.from({length:count},(_,index)=>configEntry(test,index+1,{provider:'systemone',endpoint:`http://127.0.0.1:${1300+index}/v1/systemone`,model:`dynamic-${index}`}));
    test.state.explanationSettings={enabled:true,model:'qwen38-27b:q4_k_m'};
    test.state.rubric.criteria[0].levels.splice(1,0,{id:'partial',label:'Partial',points:6,description:''});get('submission-text').value='Synthetic dynamic explanation';
    const grading=test.grade(),sent=JSON.parse(pending().options.body);
    const entries=sent.models.map(settings=>({label:settings.model,ok:true,result:{...result,model:settings.model,provider:settings.provider,total:6,criteria:[{...result.criteria[0],level_id:'partial',level_label:'Partial',points:6,confidence:0.9}]}}));
    respond(pending(),{results:entries,comparison:[{id:'c',disagrees:count===1?null:false,comparison_receipt:count===1?null:'dynamic-receipt'}]});await grading;
    assert.equal(requests.length,count===1?1:2);
    if(count===1){assert.equal(test.state.explanations.length,0);continue;}
    const body=JSON.parse(pending().options.body);assert.equal(body.decisions.length,count);assert.deepEqual(body.decisions.map(row=>row.model),sent.models.map(model=>model.model));assert.deepEqual(body.decisions.map(row=>row.endpoint),sent.models.map(model=>model.endpoint));
    const instructor=get('results-content').querySelectorAll('input')[0];instructor.value='8';instructor.listeners.get('input')();
    respond(pending(),{status:'ok',mode:'agreement',verdict:'agree',criterion_id:'c',explanation:'Synthetic local review.',evidence:[],model:'qwen38-27b:q4_k_m'});await tick();
    assert.equal(test.state.overrides.c.points,8);assert.ok(test.exportResult().results.every(entry=>entry.result.criteria[0].points===6));
  }
  // Four requested models keep stable identities through success, failures, review, and exports.
  {
    const {test,get,result,pending,requests}=setup();
    test.state.models[2]=configEntry(test,3,{provider:'systemone',endpoint:'http://127.0.0.1:1236/v1/systemone',model:'startlux-test'});
    test.state.models[3]=configEntry(test,4,{provider:'systemone',endpoint:'http://127.0.0.1:1238/v1/systemone',model:'d3-test'});
    test.state.models.forEach(entry=>entry.settings.scoring_mode='strict_review');
    test.renderSettings();
    test.bindEvents(); get('submission-text').value='Synthetic four-model comparison';
    const grading=test.grade(), payload=JSON.parse(pending().options.body);
    assert.deepEqual(payload.models.map(model=>model.model),['local-test','clef-test','startlux-test','d3-test']);
    assert.ok(payload.models.every(model=>model.scoring_mode==='strict_review' && model.confidence_threshold===0.65 && model.max_input_chars===24000));
    const scored={...result,model:'local-test',criteria:[{...result.criteria[0],level_id:'yes',level_label:'Demonstrated',confidence:0.8,needs_review:false,probabilities:[]}]};
    const withheld={...scored,model:'d3-test',total:null,status:'review_required',criteria:[{...scored.criteria[0],points:null,level_id:null,level_label:null,needs_review:true,top_choice:{label:'Demonstrated',points:9,probability:0.55}}]};
    const entries=[{label:'Model A · Laya',ok:true,result:scored},{label:'Model B · Clef',ok:false,error:'Synthetic offline server'},{label:'Model C · StartLux',ok:true,result:{...scored,model:'startlux-test'}},{label:'Model D · D3',ok:true,result:withheld}];
    pending().resolve({ok:true,json:async()=>({results:entries,comparison:[{id:'c',available_models:2,model_points:[9,null,9,null],model_level_ids:['yes',null,'yes',null],disagrees:false}],warnings:[]})});
    await grading;
    const content=get('results-content'), blocks=content.querySelectorAll('.model-decision-block');
    assert.equal(content.querySelectorAll('.comparison-total').length,4); assert.equal(blocks.length,4);
    assert.match(blocks[1].textContent,/Synthetic offline server/); assert.equal(blocks[1].querySelectorAll('.criterion-score').length,0);
    assert.match(blocks[3].textContent,/Review required/); assert.equal(blocks[3].querySelectorAll('.criterion-score').length,0);
    assert.match(content.textContent,/2 available models agree/); assert.doesNotMatch(content.textContent,/All 4 models agree/);
    assert.match(get('grading-status').textContent,/3 \/ 4 models returned/);
    assert.equal(content.querySelectorAll('input')[0].value,''); assert.equal(get('final-total').textContent,'—');
    const exported=test.exportResult(), rows=test.resultCsvRows(exported), headers=rows[0], row=rows[1];
    assert.equal(exported.results.length,4); assert.ok(rows.every(item=>item.length===headers.length));
    for (const [index,letter] of ['A','B','C','D'].entries()) assert.equal(row[headers.indexOf(`Model ${letter}`)],entries[index].label);
    assert.equal(row[headers.indexOf('Model C score')],9); assert.equal(row[headers.indexOf('Model B score')],''); assert.equal(row[headers.indexOf('Model D score')],'');
    assert.equal(row[headers.indexOf('Model B status')],'unavailable'); assert.equal(row[headers.indexOf('Model D status')],'review_required');
    assert.equal(row[headers.indexOf('Model D top preference probability')],0.55);
    assert.equal(rows[2][headers.indexOf('Model D score')],''); assert.equal(exported.review.total,null);
    get('model-m3').value='updated-startlux';get('model-m3').listeners.get('input')();
    assert.equal(test.state.result,null);assert.throws(()=>test.exportResult(),/Compare a submission/);
    // Probes are sequential because the local backend serializes connection checks.
    const before=requests.length, probing=get('probe-model').listeners.get('click')();
    for (let i=0;i<4;i++) {
      assert.equal(requests.length,before+i+1); assert.equal(test.activeRequests.size,1);
      assert.equal(pending().url,'/api/probe');
      assert.equal(JSON.parse(pending().options.body).settings.model,['local-test','clef-test','updated-startlux','d3-test'][i]);
      pending().resolve({ok:i!==1,json:async()=>i===1?{error:'Synthetic connection failure'}:{ok:true,message:'Ready'}});
      await new Promise(resolve=>setImmediate(resolve));
    }
    await probing;assert.equal(test.activeRequests.size,0);assert.equal(test.state.busy,false);
    assert.match(get('model-status').textContent,/local-test: Ready.*clef-test: Synthetic connection failure.*updated-startlux: Ready.*d3-test: Ready/);
    // Editing the fourth model while a grade is in flight invalidates the whole comparison.
    const stale=test.grade();get('endpoint-m4').value='http://127.0.0.1:1240/v1/systemone';get('endpoint-m4').listeners.get('input')();
    pending().resolve({ok:true,json:async()=>({results:entries,comparison:[]})});await stale;
    assert.equal(test.state.comparison,null);assert.equal(test.state.result,null);
    // No available models produces visible errors and no instructor baseline or export.
    const failed=test.grade();pending().resolve({ok:true,json:async()=>({results:entries.map(entry=>({label:entry.label,ok:false,error:'Synthetic unavailable'})),comparison:[]})});await failed;
    assert.equal(content.querySelectorAll('.comparison-total').length,4);assert.equal(test.state.result,null);
    assert.equal(get('result-actions').hidden,true);assert.match(content.textContent,/No local model returned a grade/);
    const requestCount=requests.length;get('model-m4').value='';get('model-m4').listeners.get('input')();
    await get('probe-model').listeners.get('click')();assert.equal(requests.length,requestCount);
    assert.match(get('model-status').textContent,/Every model needs its local model identifier/);
    await test.grade();assert.equal(requests.length,requestCount);assert.match(get('global-status').textContent,/Every model needs its local model identifier/);

  }
  {
    const {test,get,requests}=setup(); test.state.initialized=false;
    const defaults={rubric:test.state.rubric,comparison_models:test.currentModels(),explanation_settings:test.state.explanationSettings};
    const initializing=test.init();respond(requests.find(r=>r.url==='/api/defaults'),defaults);respond(requests.find(r=>r.url==='/api/model-catalog'),{models:[]});respond(requests.find(r=>r.url==='/api/workspace'),{error:'Malformed saved configuration'},false);await initializing;
    assert.equal(test.state.initialized,true);assert.equal(test.state.models.length,2);
    assert.match(get('workspace-status').textContent,/Saved configuration could not be loaded/);
    assert.equal(requests.length,3,'Failed saved configuration never causes an automatic write');
  }
  // Agreement explanations retain their eligibility, retry, and lifecycle protections.
  for (const variant of ['consensus','custom-model','failure','withheld','full-credit','disabled','stale','next-student','pagehide','disable-after-start','unavailable','retry-success','retry-expired','retry-stale','retry-next','retry-pagehide','missing-proof']) {
    const {test,get,result,pending,requests,windowEvents}=setup();
    test.state.models[2]=configEntry(test,3,{provider:'systemone',model:'startlux-test',endpoint:'http://127.0.0.1:1236/v1/systemone'});test.state.models[3]=configEntry(test,4,{provider:'systemone',model:'d3-test',endpoint:'http://127.0.0.1:1238/v1/systemone'});
    const explanationModel=variant==='custom-model'?'deepseek-r1:70b':'qwen38-27b:q4_k_m';
    test.state.explanationSettings={enabled:variant!=='disabled',model:'qwen38-27b:q4_k_m'};
    test.state.rubric.criteria[0].levels.splice(1,0,{id:'partial',label:'Partial',description:'Some major requirements missing.',points:6});
    const points=variant==='full-credit'?9:6, level=points===9?'yes':'partial';
    const retryCase=variant.startsWith('retry-');
    const queueCase=['stale','next-student','pagehide','disable-after-start'].includes(variant);
    if (queueCase) test.state.rubric.criteria.push({...test.state.rubric.criteria[0],id:'c2',name:'Second synthetic criterion'});
    const scored={...result,total:points,model:'local-test',provider:'laya',criteria:[{...result.criteria[0],points,level_id:level,level_label:'Partial',confidence:0.9,needs_review:false,probabilities:[]}]};
    if (queueCase) scored.criteria.push({...scored.criteria[0],id:'c2',name:'Second synthetic criterion'});
    const entries=['Laya','Clef','StartLux','D3'].map((label,index)=>({label,ok:true,result:{...scored,model:label,provider:index?'systemone':'laya'}}));
    if (variant==='disagreement') entries[3].result={...scored,criteria:[{...scored.criteria[0],points:9,level_id:'yes'}]};
    if (variant==='failure') entries[3]={label:'D3',ok:false,error:'offline'};
    if (variant==='withheld') entries[3].result={...scored,total:null,criteria:[{...scored.criteria[0],points:null,level_id:null}]};
    test.state.assignmentDetails='Synthetic assignment context';test.state.gradingGuidance='Synthetic guidance';
    test.bindEvents();get('submission-text').value='Synthetic evidence only';
    if(variant==='custom-model'){get('explanation-model').value=explanationModel;get('explanation-model').listeners.get('input')();}
    const grading=test.grade();
    pending().resolve({ok:true,json:async()=>({results:entries,comparison:[{id:'c',disagrees:variant==='disagreement',comparison_receipt:variant==='missing-proof'?null:'signed-synthetic-receipt'},{id:'c2',disagrees:false,comparison_receipt:'signed-second-receipt'}],warnings:[]})});
    await grading;
    const eligible=retryCase || ['consensus','custom-model','stale','next-student','pagehide','disable-after-start','unavailable'].includes(variant);
    assert.equal(requests.length,eligible?2:1,`${variant}: no explanation request for ineligible comparison`);
    if (variant==='missing-proof') {assert.equal(test.state.explanations[0].status,'ineligible');assert.match(get('results-content').textContent,/Grade again/);assert.doesNotMatch(get('results-content').textContent,/authorization/);assert.equal(get('results-content').querySelectorAll('.retry-explanation').length,0);continue;}
    if (!eligible) {assert.equal(test.state.explanations.length,0);continue;}
    assert.equal(pending().url,'/api/explain');assert.equal(test.state.busy,false,'Explanations do not block instructor review');
    const payload=JSON.parse(pending().options.body);
    assert.equal(payload.comparison_receipt,'signed-synthetic-receipt');assert.equal(payload.criterion_id,'c');
    assert.equal(payload.text,'Synthetic evidence only');assert.equal(payload.assignment_details,'Synthetic assignment context');assert.equal(payload.grading_guidance,'Synthetic guidance');
    assert.equal(payload.settings.model,explanationModel);assert.equal(payload.decisions.length,4);
    assert.ok(payload.decisions.every(row=>row.ok && row.level_id==='partial' && row.points===6));
    assert.equal(payload.decisions[0].endpoint,null);
    assert.deepEqual(payload.decisions.slice(1).map(row=>row.endpoint),['http://127.0.0.1:1235/v1/systemone','http://127.0.0.1:1236/v1/systemone','http://127.0.0.1:1238/v1/systemone']);
    assert.equal(test.exportResult().explanations[0].status,'pending');
    const content=get('results-content'), instructor=content.querySelectorAll('input')[0];
    instructor.value='9';instructor.listeners.get('input')();
    if (variant==='stale') {get('submission-text').value='Changed submission';get('submission-text').listeners.get('input')();}
    if (variant==='next-student') test.clearSubmission();
    if (variant==='disable-after-start') {get('explanation-enabled').checked=false;get('explanation-enabled').listeners.get('change')();}
    if (variant==='pagehide') {windowEvents.get('pagehide')();assert.equal(pending().options.signal.aborted,true);}
    pending().resolve({ok:variant!=='unavailable',json:async()=>variant==='unavailable'?{error:'Dedicated local Ollama unavailable'}:retryCase?{status:'unavailable',message:'The local explanation model is busy.'}:{status:'ok',criterion_id:'c',verdict:'disagree',explanation:'<script>not executed</script> The rubric deserves another instructor review.',evidence:['Synthetic evidence only'],model:explanationModel}});
    await new Promise(resolve=>setImmediate(resolve));
    if (queueCase) {assert.equal(requests.length,2,'No subsequent queued criterion runs after invalidation');assert.equal(test.state.result,null);assert.equal(test.state.explanations.length,0);assert.equal(content.children.length,0);continue;}
    assert.equal(get('final-total').textContent,'9');assert.equal(test.state.overrides.c.points,9,'Explanation completion preserves entered instructor score');
    const exported=test.exportResult(), explanation=exported.explanations[0];
    assert.ok(exported.results.every(entry=>entry.result.criteria[0].points===6),'Explanation never changes model points');
    assert.equal(exported.review.total,9);
    assert.equal(explanation.status,variant==='unavailable'||retryCase?'unavailable':'ok');
    if (variant==='unavailable') {assert.match(content.textContent,/Dedicated local Ollama unavailable/);assert.equal(content.querySelectorAll('.retry-explanation').length,1);}
    else if (retryCase) {
      const retry=content.querySelectorAll('.retry-explanation')[0]; assert.ok(retry);
      const originalRequest=JSON.parse(pending().options.body);
      retry.listeners.get('click')();retry.listeners.get('click')();
      assert.equal(requests.length,3,'Repeated retry clicks create only one request');
      assert.equal(requests.filter(request=>request.url==='/api/compare').length,1,'Retry never reruns the graders');
      assert.deepEqual(JSON.parse(pending().options.body),originalRequest,'Retry uses the original context, decisions, and receipt');
      assert.equal(content.querySelectorAll('.retry-explanation').length,0);assert.equal(test.exportResult().explanations[0].status,'pending');
      assert.equal(get('final-total').textContent,'9');
      if (variant==='retry-stale') {get('submission-text').value='Edited';get('submission-text').listeners.get('input')();}
      if (variant==='retry-next') test.clearSubmission();
      if (variant==='retry-pagehide') {windowEvents.get('pagehide')();assert.equal(pending().options.signal.aborted,true);}
      pending().resolve({ok:true,json:async()=>variant==='retry-expired'?{status:'ineligible',message:'The original comparison proof expired or no longer matches.'}:{status:'ok',criterion_id:'c',verdict:'agree',explanation:'A synthetic reason.',evidence:[],model:explanationModel}});
      await new Promise(resolve=>setImmediate(resolve));
      assert.equal(requests.length,3);
      if (['retry-stale','retry-next','retry-pagehide'].includes(variant)) {
        assert.equal(test.state.result,null);assert.equal(test.state.explanationContext,null);assert.equal(content.children.length,0);
        retry.listeners.get('click')();assert.equal(requests.length,3,'Detached stale retry buttons cannot send data');
      } else {
        assert.equal(test.state.overrides.c.points,9);assert.equal(get('final-total').textContent,'9');
        assert.ok(test.exportResult().results.every(entry=>entry.result.criteria[0].points===6));
        assert.equal(content.querySelectorAll('.retry-explanation').length,0);
        assert.match(content.textContent,variant==='retry-expired'?/Grade again/:/Agrees with the shared rating/);
        if (variant==='retry-success') assert.equal(test.exportResult().explanations[0].message,undefined,'Completed export has no stale pending message');
      }
    } else {
      assert.match(content.textContent,/Disagrees with the shared rating/);assert.match(content.textContent,/<script>not executed<\/script>/);
      assert.equal(content.querySelectorAll('script').length,0,'Explanation strings are text, never HTML');
      const rows=test.resultCsvRows(exported);assert.equal(rows[1][rows[0].indexOf('Explanation verdict')],'disagree');
      assert.equal(rows[1][rows[0].indexOf('Explanation evidence JSON')],'["Synthetic evidence only"]');
      assert.equal(rows[2][rows[0].indexOf('Explanation verdict')],'');
    }
  }
  // Disagreements use original level identities, including equal points at full credit.
  for (const variant of ['best-fit','zero-fit','uncertain','same-points','retry','stale','next','pagehide']) {
    const {test,get,result,pending,requests,windowEvents}=setup();
    test.state.models[2]=configEntry(test,3,{provider:'systemone',model:'startlux-test',endpoint:'http://127.0.0.1:1236/v1/systemone'});
    test.state.models[3]=configEntry(test,4,{provider:'systemone',model:'d3-test',endpoint:'http://127.0.0.1:1238/v1/systemone'});
    test.state.explanationSettings={enabled:true,model:'deepseek-r1:70b'};
    test.state.rubric.criteria[0].levels.splice(1,0,{id:'partial',label:'Partial',points:6,description:'Some requirements missing.'});
    if (variant==='same-points') test.state.rubric.criteria[0].levels.push({id:'alternative',label:'Other full-credit level',points:9,description:'A different rubric-defined achievement.'});
    const decisions=variant==='same-points'?[['yes',9],['alternative',9],['yes',9],['yes',9]]:[['yes',9],['partial',6],['partial',6],['partial',6]];
    const entries=['Laya','Clef','StartLux','D3'].map((label,index)=>({label,ok:true,result:{...result,model:label,provider:index?'systemone':'laya',total:decisions[index][1],criteria:[{...result.criteria[0],level_id:decisions[index][0],level_label:decisions[index][0],points:decisions[index][1],confidence:0.8,probabilities:[]}]}}));
    const originalResults=JSON.parse(JSON.stringify(entries));
    test.bindEvents();get('submission-text').value='Synthetic disagreement evidence';
    const grading=test.grade();pending().resolve({ok:true,json:async()=>({results:entries,comparison:[{id:'c',disagrees:true,comparison_receipt:'signed-disagreement'}],warnings:[]})});await grading;
    assert.equal(requests.length,2);assert.equal(pending().url,'/api/explain');
    const body=JSON.parse(pending().options.body);assert.equal(body.comparison_receipt,'signed-disagreement');
    assert.deepEqual(body.decisions.map(row=>[row.level_id,row.points]),decisions);
    assert.equal(test.exportResult().explanations[0].mode,'disagreement');
    const content=get('results-content'), instructor=content.querySelectorAll('input')[0];
    assert.match(content.textContent,/reviewing the disagreement/);assert.equal(instructor.value,'');
    instructor.value='7';instructor.listeners.get('input')();
    if (variant==='retry') {
      pending().resolve({ok:true,json:async()=>({status:'unavailable',message:'Busy local model.'})});await new Promise(resolve=>setImmediate(resolve));
      const retry=content.querySelectorAll('.retry-explanation')[0];retry.listeners.get('click')();retry.listeners.get('click')();
      assert.equal(requests.length,3);assert.deepEqual(JSON.parse(pending().options.body),body);
      instructor.value='8';instructor.listeners.get('input')();
    }
    if (variant==='stale') {get('grading-guidance').value='Updated course guidance';get('grading-guidance').listeners.get('input')({target:get('grading-guidance')});}
    if (variant==='next') test.clearSubmission();
    if (variant==='pagehide') {windowEvents.get('pagehide')();assert.equal(pending().options.signal.aborted,true);}
    const bestFit=variant==='uncertain'?null:variant==='zero-fit'?{level_id:'no',label:'Not demonstrated',points:0}:variant==='same-points'?{level_id:'alternative',label:'Other full-credit level',points:9}:{level_id:'partial',label:'Partial',points:6};
    pending().resolve({ok:true,json:async()=>({status:'ok',mode:'disagreement',verdict:bestFit?'best_fit':'uncertain',best_fit_level:bestFit,criterion_id:'c',explanation:'Synthetic explanation of the conflicting ratings.',evidence:['Synthetic disagreement evidence'],model:'deepseek-r1:70b',advisory_only:true})});
    await new Promise(resolve=>setImmediate(resolve));
    if (['stale','next','pagehide'].includes(variant)) {assert.equal(test.state.result,null);assert.equal(test.state.explanationContext,null);assert.equal(content.children.length,0);continue;}
    const expectedInstructor=variant==='retry'?8:7, exported=test.exportResult();
    assert.equal(get('final-total').textContent,String(expectedInstructor));assert.equal(exported.review.total,expectedInstructor);
    assert.equal(content.querySelectorAll('input')[0],instructor,'Advisory completion never recreates instructor fields');
    assert.deepEqual(JSON.parse(JSON.stringify(exported.results)),originalResults,'Original scores, levels, totals, and all model metadata remain unchanged');
    assert.equal(exported.explanations[0].advisory_only,true);assert.equal(exported.explanations[0].mode,'disagreement');
    assert.deepEqual(JSON.parse(JSON.stringify(exported.explanations[0].best_fit_level)),bestFit);
    assert.match(content.textContent,/Explanation only. Model and instructor scores are unchanged/);
    if (bestFit) assert.ok(content.textContent.includes(`Advisory best fit: ${bestFit.label} · ${bestFit.points} rubric points`));
    else {assert.match(content.textContent,/Uncertain · no best-fit rubric level selected/);assert.doesNotMatch(content.textContent,/Advisory best fit:/);}
    const rows=test.resultCsvRows(exported), headers=rows[0], row=rows[1];
    assert.ok(rows.every(item=>item.length===headers.length));
    assert.equal(row[headers.indexOf('Explanation mode')],'disagreement');
    assert.equal(row[headers.indexOf('Advisory best-fit level ID')],bestFit?.level_id||'');
    assert.equal(row[headers.indexOf('Advisory best-fit label')],bestFit?.label||'');
    assert.equal(row[headers.indexOf('Advisory best-fit rubric points')],bestFit?.points??'');
    assert.equal(row[headers.indexOf('Instructor score')],expectedInstructor);
    assert.equal(row[headers.indexOf('Model A score')],decisions[0][1]);assert.equal(row[headers.indexOf('Model D score')],decisions[3][1]);
    assert.equal(rows[2][headers.indexOf('Advisory best-fit rubric points')],'');
    assert.equal(requests.filter(request=>request.url==='/api/compare').length,1);
  }
  // Visual recognition is review-only until an explicit, checked append action.
  {
    const {test,get,result,pending,requests}=setup();test.bindEvents();
    const file={name:'synthetic-diagram.pdf',size:100};
    const preview={image_data_url:'data:image/png;base64,AAAA',width:120,height:80,index:1,count:3,source_label:'synthetic-diagram.pdf · page 1',warnings:['<img src=x onerror=alert(1)> is plain text']};
    const comparison={results:[{label:'A',ok:true,result},{label:'B',ok:true,result}],comparison:[]};
    test.state.result=result;test.state.comparison=comparison;test.state.overrides={c:{points:7,note:'Instructor note'}};
    get('submission-text').value='Existing synthetic response.';
    const preparing=test.prepareVision(file);
    assert.equal(pending().url,'/api/vision/prepare');assert.equal(pending().options.body.get('file'),file);assert.equal(pending().options.body.get('index'),'1');
    assert.equal(get('grade-button').disabled,true);
    respond(pending(),preview);await preparing;
    assert.equal(get('vision-image').src,preview.image_data_url);assert.equal(get('vision-preview').hidden,false);
    assert.equal(get('vision-warnings').querySelectorAll('img').length,0);assert.match(get('vision-warnings').textContent,/<img/);
    assert.equal(test.state.result,result);assert.equal(test.state.comparison,comparison);assert.equal(test.state.overrides.c.points,7);
    const analyzing=test.analyzeVision();
    assert.equal(pending().url,'/api/vision/analyze');
    assert.deepEqual(JSON.parse(pending().options.body),{image_data_url:preview.image_data_url,model:'openbmb/minicpm-v4.7:1b',detailed_crops:true,transcribe:true});
    const count=requests.length;await test.grade();assert.equal(requests.length,count,'Grading waits for the current vision request');
    respond(pending(),{model:'openbmb/minicpm-v4.7:1b',transcription:'<script>student text</script>',description:'A points to B.',uncertainties:['<img src=evil> uncertain label']});await analyzing;
    assert.equal(get('vision-transcription').value,'<script>student text</script>');assert.equal(get('vision-uncertainties').querySelectorAll('img').length,0);
    assert.equal(test.state.result,result);assert.equal(test.state.comparison,comparison);assert.equal(test.state.overrides.c.points,7,'Recognition leaves original instructor scores unchanged');
    assert.equal(get('submission-text').value,'Existing synthetic response.');
    test.appendVision();assert.equal(get('submission-text').value,'Existing synthetic response.','Unchecked analysis cannot enter grading input');
    get('vision-reviewed').checked=true;get('vision-reviewed').listeners.get('change')();assert.equal(get('vision-append').disabled,false);
    get('vision-description').value='A points to C.';get('vision-description').listeners.get('input')();assert.equal(get('vision-reviewed').checked,false,'Any edit must be reviewed again');
    get('vision-reviewed').checked=true;get('vision-reviewed').listeners.get('change')();test.appendVision();
    assert.match(get('submission-text').value,/^Existing synthetic response\./);assert.match(get('submission-text').value,/Reviewed image evidence: synthetic-diagram\.pdf · page 1/);
    assert.match(get('submission-text').value,/AI-assisted visual description \(reviewed\):\nA points to C\./);
    assert.match(get('submission-text').value,/Recognition uncertainties to consider/);assert.equal(test.state.result,null);assert.equal(test.state.comparison,null);
    const appended=get('submission-text').value;get('vision-reviewed').checked=true;test.appendVision();assert.equal(get('submission-text').value,appended,'Repeated append clicks do not duplicate evidence');
    get('vision-index').value='2';const nextPage=get('vision-show-page').listeners.get('click')();assert.equal(pending().options.body.get('index'),'2');assert.equal(test.state.vision.analysis,null);assert.equal(get('vision-transcription').value,'');
    respond(pending(),{...preview,index:2,source_label:'synthetic-diagram.pdf · page 2'});await nextPage;
    assert.equal(get('vision-index').value,'2');assert.equal(get('submission-text').value,appended,'Changing the preview leaves already reviewed evidence alone');
    const settings=test.workspaceSnapshot();assert.equal(settings.vision_settings.model,'openbmb/minicpm-v4.7:1b');assert.doesNotMatch(JSON.stringify(settings),/synthetic-diagram|A points|image_data_url|student text|uncertain label/);
    assert.throws(()=>test.normalizeWorkspace({...settings,vision_settings:{...settings.vision_settings,image_data_url:preview.image_data_url}}),/installed local Ollama vision model/);
    const html=fs.readFileSync(require('node:path').join(__dirname,'../grader/static/index.html'),'utf8');
    assert.match(html,/<details id="vision-panel" class="vision-panel">/,'Visual content starts collapsed');
    assert.match(html,/Windows, Linux, and macOS setup/);assert.doesNotMatch(html,/images are ignored|TEXT ONLY/);
  }
  // Late prepare/analyze responses cannot restore a cleared or replaced student's image.
  for(const phase of ['prepare','analyze'])for(const action of ['next','pagehide','replace','model-change','detail-change','reset']) {
    const {test,get,pending,windowEvents}=setup();test.bindEvents();
    test.state.defaults={rubric:test.state.rubric,models:test.state.models,explanationSettings:test.state.explanationSettings,visionSettings:{model:'openbmb/minicpm-v4.7:1b'}};
    const file={name:'synthetic.png',size:10},preview={image_data_url:'data:image/png;base64,AAAA',width:20,height:20,index:1,count:1,source_label:'synthetic.png',warnings:[]};
    let request=test.prepareVision(file);
    if(phase==='analyze'){respond(pending(),preview);await request;request=test.analyzeVision();}
    const oldRequest=pending();let replacement;
    if(action==='next')test.clearSubmission();
    if(action==='pagehide')windowEvents.get('pagehide')();
    if(action==='replace')replacement=test.prepareVision({name:'replacement.webp',size:20});
    if(action==='model-change'){get('vision-model').value='qwen3.5:2b-q4_K_M';get('vision-model').listeners.get('input')();}
    if(action==='detail-change'){get('vision-detailed-crops').checked=false;get('vision-detailed-crops').listeners.get('change')();}
    if(action==='reset')get('forget-workspace').listeners.get('click')();
    assert.equal(oldRequest.options.signal.aborted,true,`${phase}/${action}: old local request is aborted`);
    respond(oldRequest,phase==='prepare'?preview:{model:'old',transcription:'STALE IMAGE TEXT',description:'STALE DIAGRAM',uncertainties:['STALE']});await request;
    assert.equal(test.state.vision.analysis,null);assert.equal(get('vision-transcription').value,'');assert.equal(get('vision-description').value,'');assert.equal(get('submission-text').value,'');
    if(['next','pagehide','reset'].includes(action)){assert.equal(test.state.vision.file,null);assert.equal(test.state.vision.prepared,null);assert.equal(get('vision-image').src,'');assert.equal(get('vision-status').textContent,'');}
    if(action==='replace'){assert.equal(test.state.vision.busy,true,'Old finally does not reset a newer request');respond(pending(),{...preview,source_label:'replacement.webp'});await replacement;assert.match(get('vision-source').textContent,/replacement.webp/);}
  }
  {
    const {test,get,pending,requests}=setup();test.bindEvents();
    await test.prepareVision({name:'too-large.png',size:15*1024*1024+1});await test.prepareVision({name:'unsafe.svg',size:100});assert.equal(requests.length,0);
    get('vision-file').files=[{name:'a.png',size:10},{name:'b.png',size:10}];await get('vision-file').listeners.get('change')();assert.equal(requests.length,0);assert.match(get('vision-status').textContent,/one image or document/);
    for(const unsafe of ['https://remote.example/image.png','data:image/svg+xml;base64,AAAA','javascript:alert(1)']){
      const preparing=test.prepareVision({name:'synthetic.png',size:10});respond(pending(),{image_data_url:unsafe,index:1,count:1});await preparing;
      assert.equal(test.state.vision.prepared,null);assert.equal(get('vision-image').src,'');assert.match(get('vision-status').textContent,/invalid image preview/);
    }
    const preparing=test.prepareVision({name:'synthetic.png',size:10});respond(pending(),{image_data_url:'data:image/png;base64,AAAA',index:1,count:1,width:20,height:20,warnings:[]});await preparing;
    const analyzing=test.analyzeVision();respond(pending(),{error:'Install the selected local vision model first.'},false);await analyzing;
    assert.equal(test.state.vision.analysis,null);assert.equal(get('submission-text').value,'');assert.match(get('vision-status').textContent,/Install/);assert.equal(get('vision-append').disabled,true);
    get('vision-model').value='https://remote.example/model';get('vision-model').listeners.get('input')();const count=requests.length;await test.analyzeVision();assert.equal(requests.length,count);
    assert.match(get('vision-status').textContent,/installed local vision model/);
  }
  // One normal document upload extracts text and reuses its File for detected visuals.
  for(const extension of ['pdf','docx']){
    const {test,get,pending,requests}=setup();test.bindEvents();
    const file={name:`mixed.${extension}`,size:200};
    const visualItems=[{index:2,label:'Page / image 2: synthetic chart'},{index:4,label:'<script>Page / image 4</script>'}];
    get('submission-file').files=[file];
    const importing=get('submission-file').listeners.get('change')();
    assert.equal(pending().url,'/api/extract');assert.equal(pending().options.body.get('file'),file);assert.equal(pending().options.body.get('detect_visuals'),'true');
    respond(pending(),{text:'Extracted synthetic document text.',filename:file.name,visuals:visualItems,visual_count:5,warnings:[]});await tick();
    assert.equal(requests.length,2);assert.equal(pending().url,'/api/vision/prepare');assert.equal(pending().options.body.get('file'),file,'Automatic preview reuses the original upload object');assert.equal(pending().options.body.get('index'),'2');
    respond(pending(),{image_data_url:'data:image/png;base64,AAAA',width:20,height:20,index:2,count:5,source_label:file.name+' page / image 2',warnings:[]});await importing;
    assert.equal(get('submission-text').value,'Extracted synthetic document text.');assert.equal(get('vision-panel').open,true);assert.equal(get('vision-detected').hidden,false);assert.equal(test.state.vision.file,file);
    assert.equal(get('vision-detected-select').children.length,2);assert.equal(get('vision-detected-select').querySelectorAll('script').length,0);
    assert.equal(requests.filter(request=>request.url==='/api/vision/analyze').length,0,'Detection never silently starts inference');
    get('vision-detected-select').value='4';const switching=get('vision-detected-select').listeners.get('change')();assert.equal(pending().options.body.get('file'),file);assert.equal(pending().options.body.get('index'),'4');
    respond(pending(),{image_data_url:'data:image/png;base64,AAAA',width:20,height:20,index:4,count:5,source_label:file.name+' page / image 4',warnings:[]});await switching;
    assert.equal(test.state.vision.detected.length,2);assert.equal(get('vision-detected-select').value,'4');
    const settings=test.workspaceSnapshot();assert.equal(settings.vision_settings.auto_detect,true);assert.doesNotMatch(JSON.stringify(settings),/mixed\.|synthetic chart|Extracted synthetic|image_data_url/);
    test.clearSubmission();assert.equal(test.state.vision.detected.length,0);assert.equal(get('vision-detected-select').children.length,0);assert.equal(get('vision-detected').hidden,true);assert.equal(test.state.vision.file,null);
  }
  {
    const {test,get,pending,requests,result}=setup();test.bindEvents();test.renderSettings();
    get('vision-auto-detect').checked=false;get('vision-auto-detect').listeners.get('change')();assert.equal(test.state.visionSettings.auto_detect,false);assert.equal(test.workspaceSnapshot().vision_settings.auto_detect,false);
    const old=test.workspaceSnapshot();delete old.vision_settings.auto_detect;assert.equal(test.normalizeWorkspace(old).vision_settings.auto_detect,true,'Older model-only configuration enables the new default');
    assert.throws(()=>test.normalizeWorkspace({...old,vision_settings:{...old.vision_settings,auto_detect:'yes'}}),/enabled or disabled/);
    const importing=test.extractFile({name:'opt-out.pdf',size:100},'submission');assert.equal(pending().options.body.get('detect_visuals'),'false');respond(pending(),{text:'Text only by choice.',visuals:[{index:1,label:'Must not prepare'}],warnings:[]});await importing;
    assert.equal(requests.length,1);assert.equal(test.state.vision.file,null);assert.equal(get('submission-text').value,'Text only by choice.');
    test.state.result=result;const revision=test.state.revision;
    const file={name:'standalone.webp',size:100};get('submission-file').files=[file];const imageImport=get('submission-file').listeners.get('change')();
    assert.equal(pending().url,'/api/vision/prepare','Standalone image uses the same main picker');respond(pending(),{image_data_url:'data:image/png;base64,AAAA',width:20,height:20,index:1,count:1,source_label:file.name,warnings:[]});await imageImport;
    assert.equal(test.state.result,result);assert.equal(test.state.revision,revision);assert.equal(get('submission-text').value,'Text only by choice.','An image draft does not overwrite existing text');assert.equal(get('vision-panel').open,true);
    assert.equal(requests.filter(request=>request.url==='/api/extract').length,1);
  }
  for(const action of ['edit','next','pagehide','opt-out']){
    const {test,get,pending,requests,windowEvents}=setup();test.bindEvents();
    const importing=test.extractFile({name:'stale.pdf',size:100},'submission');
    if(action==='edit'){get('submission-text').value='Newer text';get('submission-text').listeners.get('input')();}
    if(action==='next')test.clearSubmission();
    if(action==='pagehide')windowEvents.get('pagehide')();
    if(action==='opt-out'){get('vision-auto-detect').checked=false;get('vision-auto-detect').listeners.get('change')();}
    respond(pending(),{text:'Old extracted text',visuals:[{index:1,label:'Stale chart'}],warnings:[]});await importing;
    assert.equal(requests.length,1,`${action}: no late automatic preview starts`);assert.equal(test.state.vision.file,null);assert.equal(test.state.vision.detected.length,0);
    if(action==='edit')assert.equal(get('submission-text').value,'Newer text');
    if(['next','pagehide'].includes(action))assert.equal(get('submission-text').value,'');
  }
  // PDF native text is never sent to vision or overwritten by its recognition draft.
  for(const nativeImported of [true,false]) {
    const {test,get,pending,result}=setup();test.bindEvents();
    const file={name:'synthetic-vector.pdf',size:200};
    const native='SELECT * FROM teams WHERE team_id = ?;  \n\n';
    const preview={image_data_url:'data:image/png;base64,AAAA',width:2200,height:1800,index:2,count:3,image_index:0,image_count:0,kind:'pdf_vectors',native_text:native,source_label:'synthetic-vector.pdf · page 2',warnings:[]};
    get('submission-text').value=nativeImported?native:'Unchanged original text.  \n';
    const originalText=get('submission-text').value;
    test.state.result=result;test.state.overrides={c:{points:8,note:'Keep this score'}};const revision=test.state.revision;
    const preparing=test.prepareVision(file,2,[],0,nativeImported);respond(pending(),preview);await preparing;
    assert.equal(get('vision-native-text').value,native);assert.equal(get('vision-native').hidden,false);
    const analyzing=test.analyzeVision(),payload=JSON.parse(pending().options.body);
    assert.equal(payload.transcribe,false);assert.equal(payload.detailed_crops,true);assert.equal(payload.native_text,undefined);assert.doesNotMatch(JSON.stringify(payload),/SELECT|team_id/);
    const view={label:'Overview',bounds:[0,0,2200,1800],transcription:'WRONG OCR 7',description:'Two linked nodes.',uncertainties:[]};
    respond(pending(),{model:'local',transcription:'WRONG OCR 7',description:'Two linked nodes.',uncertainties:[],views:[view],view_count:1});await analyzing;
    assert.equal(get('vision-transcription-field').hidden,true);assert.equal(get('vision-transcription').value,'');
    assert.equal(get('submission-text').value,originalText);assert.equal(test.state.result,result);assert.equal(test.state.overrides.c.points,8);assert.equal(test.state.revision,revision);
    assert.doesNotMatch(get('vision-view-results').textContent,/WRONG OCR/);assert.equal(get('vision-view-results').children.length,1);
    get('vision-reviewed').checked=true;test.appendVision();
    const appended=get('submission-text').value;
    assert.ok(appended.startsWith(originalText+'\n\n'),'Appending preserves every character of the existing native text, including trailing whitespace');
    assert.equal(appended.split('team_id = ?').length-1,1,'Native text appears exactly once');
    assert.doesNotMatch(appended,/WRONG OCR/);assert.match(appended,/Two linked nodes/);
    if(!nativeImported){assert.match(appended,/Native PDF text \(verbatim\):/);assert.ok(appended.includes(native));}
    else assert.doesNotMatch(appended,/Native PDF text \(verbatim\):/);
    assert.equal(test.state.result,null,'Only explicit reviewed append invalidates outdated grades');
    test.clearSubmission();assert.equal(get('vision-native-text').value,'');assert.equal(get('vision-native').hidden,true);assert.equal(get('vision-view-results').children.length,0);assert.equal(test.state.vision.nativeImported,false);
  }
  // Two embedded images on one PDF page remain distinct through automatic/manual selection.
  {
    const {test,get,pending}=setup();test.bindEvents();
    const file={name:'synthetic-multi-image.pdf',size:300};
    const visuals=[{index:2,image_index:1,kind:'pdf_image',label:'Page 2 · image 1'},{index:2,image_index:2,kind:'pdf_image',label:'Page 2 · image 2'},{index:3,image_index:0,kind:'pdf_vectors',label:'Page 3 · diagram'}];
    const importing=test.extractFile(file,'submission');
    respond(pending(),{text:'Native SQL: SELECT ?;\n',filename:file.name,visuals,warnings:[]});await tick();
    assert.equal(pending().options.body.get('index'),'2');assert.equal(pending().options.body.get('image_index'),'1');
    const preview={image_data_url:'data:image/png;base64,AAAA',width:1800,height:1000,index:2,count:3,image_index:1,image_count:2,kind:'pdf_image',native_text:'Native SQL: SELECT ?;',source_label:'Page 2 · image 1',warnings:[]};
    respond(pending(),preview);await importing;
    assert.equal(get('vision-detected-select').value,'2:1');assert.equal(get('vision-image-index').value,'1');assert.equal(get('vision-image-controls').hidden,false);
    assert.equal(get('vision-image-index').children.length,3);assert.equal(test.state.vision.nativeImported,true);
    get('vision-detected-select').value='2:2';const choosing=get('vision-detected-select').listeners.get('change')();
    assert.equal(pending().options.body.get('index'),'2');assert.equal(pending().options.body.get('image_index'),'2');assert.equal(pending().options.body.get('file'),file);
    respond(pending(),{...preview,image_index:2,source_label:'Page 2 · image 2'});await choosing;
    assert.equal(get('vision-image-index').value,'2');assert.equal(test.state.vision.nativeImported,true);
    get('vision-image-index').value='0';const page=get('vision-image-index').listeners.get('change')();
    assert.equal(pending().options.body.get('image_index'),'0');respond(pending(),{...preview,image_index:0,kind:'pdf_vectors'});await page;
    assert.equal(get('submission-text').value,'Native SQL: SELECT ?;\n');
    get('vision-detailed-crops').checked=false;get('vision-detailed-crops').listeners.get('change')();
    const settings=test.workspaceSnapshot();assert.equal(settings.vision_settings.detailed_crops,false);
    const old=JSON.parse(JSON.stringify(settings));delete old.vision_settings.detailed_crops;assert.equal(test.normalizeWorkspace(old).vision_settings.detailed_crops,true);
    assert.throws(()=>test.normalizeWorkspace({...settings,vision_settings:{...settings.vision_settings,detailed_crops:'yes'}}),/enabled or disabled/);
    test.applyWorkspace({...settings,vision_settings:{...settings.vision_settings,detailed_crops:true}});test.renderSettings();assert.equal(get('vision-detailed-crops').checked,true);
    const analyzing=test.analyzeVision();respond(pending(),{model:'local',transcription:'',description:'A diagram.',uncertainties:[],views:[{label:'<script>Crop 1</script>',bounds:[0,0,900,600],transcription:'',description:'<img src=x>',uncertainties:[]}],view_count:1});await analyzing;
    assert.equal(get('vision-view-results').querySelectorAll('script').length,0);assert.equal(get('vision-view-results').querySelectorAll('img').length,0);assert.match(get('vision-view-results').textContent,/<img src=x>/);
  }
  // Failed crop content is never evidence; successful views remain drafts with explicit partial status.
  for(const invalidPartial of [false,true]) {
    const {test,get,pending,result}=setup();test.bindEvents();
    get('submission-text').value='Existing native text.';test.state.result=result;
    test.state.overrides={c:{points:8,note:'Keep'}};const revision=test.state.revision;
    const preparing=test.prepareVision({name:'synthetic.png',size:200},1);
    respond(pending(),{image_data_url:'data:image/png;base64,AAAA',width:1000,height:700,index:1,count:1,source_label:'Synthetic',warnings:[]});await preparing;
    const analyzing=test.analyzeVision();
    const warning='Recognition failed; this is not missing student work.';
    respond(pending(),{model:'local',transcription:'Completed text',description:'A visible node.',uncertainties:[warning],incomplete_views:1,views:[
      {label:'Whole image',bounds:[0,0,1000,700],transcription:'Completed text',description:'A visible node.',uncertainties:[],status:'complete'},
      {label:'Detail',bounds:[0,0,560,400],transcription:invalidPartial?'Unvalidated content':'',description:'',uncertainties:[warning],status:'incomplete',error:warning}
    ]});await analyzing;
    assert.equal(get('submission-text').value,'Existing native text.');assert.equal(test.state.result,result);assert.equal(test.state.overrides.c.points,8);assert.equal(test.state.revision,revision);
    if(invalidPartial){assert.equal(test.state.vision.analysis,null);assert.match(get('vision-status').textContent,/invalid partial/);}
    else {assert.match(get('vision-status').textContent,/Partial reading: 1 of 2/);assert.match(get('vision-view-results').textContent,/Could not read this view/);assert.equal(get('vision-transcription').value,'Completed text');assert.equal(get('vision-reviewed').checked,false);}
  }
  console.log('PASS: native PDF text and SQL preserved verbatim; separate embedded image selection and local crop drafts; partial-view failures explicit; persisted detail settings, stale-request cancellation, safe rendering and unchanged scores during recognition; existing upload/rubric/model/grading/explanation/privacy checks.');
})().catch(error=>{console.error(error);process.exitCode=1;});
