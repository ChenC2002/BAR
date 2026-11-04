"use strict";
(() => {
  const data = window.BAR_DEMO;
  const $ = (id) => document.getElementById(id);
  const escape = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const icon = (name) => `<i data-lucide="${name}"></i>`;
  const icons = () => window.lucide?.createIcons();
  if (!data?.cases?.length) { $("load-error").hidden = false; icons(); return; }
  const saved = new URLSearchParams(location.hash.slice(1));
  const choose = (key, allowed, fallback) => allowed.includes(saved.get(key)) ? saved.get(key) : fallback;
  const state = {
    case:Math.max(0, data.cases.findIndex(c => c.id===saved.get("case"))),
    horizon:Number(choose("horizon", ["30","90","365"], "90")),
    budget:choose("budget", ["auto","2","6","12"], "auto"),
    view:choose("view", ["explorer","experiment"], "explorer"),
    detail:choose("detail", ["trace","sources","inputs"], "trace"),
    step:0, timer:null
  };
  const output = () => data.cases[state.case].outputs[`${state.horizon}:${state.budget}`];
  const percent = (x, digits=2) => x == null ? "--" : x>0 && x<0.0001 ? "<0.01%" : `${(x*100).toFixed(digits)}%`;
  const number = (x) => Number(x).toLocaleString("en-US");
  const label = (id) => output().nodes.find((n) => n.node_id === id)?.name || id;
  const relation = (text) => text.replaceAll("_", " ");
  const reasonLabels = { plan_complete:"Plan complete", budget_exhausted:"Budget exhausted", no_anchors:"EHR fallback: no linked anchors", quality_stagnation:"Quality stopping rule", no_actions:"No remaining action", evidence_cap:"Evidence limit reached", policy_error:"Policy unavailable" };

  function stop() {
    if (state.timer) clearInterval(state.timer);
    state.timer = null;
    $("play").innerHTML = icon("play");
    $("play").title = "Replay trace";
    $("play").setAttribute("aria-label", "Replay trace");
  }
  function switchEpisode() { stop(); state.step = output().trace.steps.length; render(); }
  function render() {
    const current = data.cases[state.case], result = output(), trace = result.trace;
    $("budget").value = state.budget;
    for (const key of ["horizon", "view", "detail"]) {
      document.querySelectorAll(`[data-${key}]`).forEach(b => {
        const active = b.dataset[key]===String(state[key]);
        b.classList.toggle("active", active);
        b.setAttribute("aria-pressed", String(active));
      });
    }
    $("explorer-view").hidden = state.view!=="explorer";
    $("experiment-view").hidden = state.view!=="experiment";
    history.replaceState(null, "", "#" + new URLSearchParams({case:current.id,horizon:state.horizon,budget:state.budget,view:state.view,detail:state.detail,step:state.step}));
    $("case-list").innerHTML = data.cases.map((c,i) => `<button class="case-option ${i===state.case?"active":""}" data-case="${i}" aria-pressed="${i===state.case}">${icon(i===0?"activity":i===1?"heart-pulse":"stethoscope")}<span>${escape(c.name)}</span></button>`).join("");
    document.querySelectorAll("[data-case]").forEach((b) => b.onclick = () => { state.case = Number(b.dataset.case); switchEpisode(); });
    $("diagnoses").innerHTML = current.diagnoses.map(d => `<li>${escape(d)}</li>`).join("");
    $("history").textContent = current.history.length ? current.history.join(" · ") : "None";
    $("clinical-section").hidden = !current.clinical;
    $("clinical").innerHTML = current.clinical ? `${escape(current.clinical.age)} years · ${escape(current.clinical.sex)}<br>Creatinine ${escape(current.clinical.creatinine)} mg/dL` : "";
    $("risk").textContent = percent(result.probability);
    $("risk").title = `Recorded probability: ${result.probability}`;
    $("baseline").textContent = percent(trace.frozen_ehr_probability);
    $("baseline").title = `Recorded probability: ${trace.frozen_ehr_probability}`;
    $("risk-source").textContent = trace.fallback ? "EHR fallback" : "Evidence-aware score";
    $("target-name").textContent = label(data.target);
    $("target-horizon").textContent = `${state.horizon}-day horizon`;
    const spent = trace.steps.slice(0,state.step).reduce((s,t) => s+t.cost, 0);
    $("cost").innerHTML = `${spent.toFixed(1)} <small>/ ${trace.budget.toFixed(1)}</small>`;
    $("budget-fill").style.width = `${Math.min(100,spent/trace.budget*100)}%`;
    $("citation-count").textContent = result.citations.length;
    $("outcome").textContent = trace.fallback ? "Fallback" : "Source records";
    $("source-count").textContent = result.citations.length;
    $("step-slider").max = trace.steps.length;
    $("step-slider").value = state.step;
    $("step-slider").disabled = !trace.steps.length;
    $("step-label").textContent = `Step ${state.step} / ${trace.steps.length}`;
    $("previous").disabled = state.step===0;
    $("next").disabled = state.step===trace.steps.length;
    $("play").disabled = !trace.steps.length;
    $("trace-content").innerHTML = `<div class="trace-start">${icon("git-branch")} Initial plan · ${trace.steps.length} attempts</div>` + trace.steps.map((step,i) => {
      const edge = result.edges.find(e => e.edge_id===step.selected[0]);
      const title = edge ? `${label(edge.head)} → ${relation(edge.relation)} → ${label(edge.tail)}` : "No evidence retained";
      return `<button class="trace-item ${state.step===i+1?"active":""} ${state.step<i+1?"future":""}" data-step="${i+1}"><div class="trace-top"><span class="step-number">${i+1}</span><span>${step.query.kind==="guide"?"GUIDE":"EXPAND"}</span><span>${step.cost.toFixed(1)} units</span></div><div class="trace-title">${escape(title)}</div><div class="trace-bottom ${step.accepted?"":"rejected"}">${icon(step.accepted?"circle-check":"circle-x")}${step.accepted?"Accepted":"Rejected"}</div></button>`;
    }).join("") + (state.step===trace.steps.length ? `<div class="terminal">${icon(trace.fallback?"corner-down-left":"flag")}${escape(reasonLabels[trace.terminal_reason] || trace.terminal_reason)}</div>` : "");
    document.querySelectorAll("[data-step]").forEach(b => b.onclick = () => { stop(); state.step=Number(b.dataset.step); render(); });
    $("sources-content").innerHTML = result.citations.length ? result.citations.map((edge,i) => `<article class="source-item"><div class="source-title">${i+1}. ${escape(label(edge.head))} → ${escape(label(edge.tail))}</div><div>${escape(relation(edge.relation))}</div><div class="source-meta"><span>Support ${edge.support_score.toFixed(3)}</span><span>${edge.provenance.length} source</span></div>${edge.provenance.map(p => `<div>${escape(p.title)}</div><div class="source-record">${escape(p.source_id)}</div>`).join("")}</article>`).join("") : '<div class="empty-sources">No citations returned.</div>';
    $("trace-content").hidden = state.detail!=="trace";
    $("sources-content").hidden = state.detail!=="sources";
    $("inputs-content").hidden = state.detail!=="inputs";
    const inputs = current.inputs;
    const labNames = {creatinine:["Creatinine","mg/dL"],urea_nitrogen:["Urea nitrogen","mg/dL"],potassium:["Potassium","mEq/L"],hemoglobin:["Hemoglobin","g/dL"]};
    $("inputs-content").innerHTML = inputs ? `<h3>Example inputs</h3><dl class="input-values">${dl([["Age",inputs.age],["Sex",inputs.male?"Male":"Female"],["Stay",`${inputs.stay_days} days`],["Prior visits",inputs.prior_visits],["Current ICD-9",inputs.codes.join(", ")],["History ICD-9",inputs.history.join(", ") || "None"]])}</dl><h3 class="lab-heading">Pre-discharge labs</h3>${Object.entries(inputs.labs).map(([name,values])=>`<div class="input-lab"><strong>${escape(labNames[name]?.[0] || name)}</strong><span>${escape(labNames[name]?.[1] || "")}</span><dl>${dl([["Last",values[0]],["Min",values[1]],["Max",values[2]]])}</dl></div>`).join("")}<p class="input-note">Other measurements: missing</p>` : '<div class="empty-sources">Input details unavailable.</div>';
    if (state.view==="explorer") drawGraph();
    icons();
  }
  function wrap(text, length) {
    const words = String(text).split(/\s+/), lines=[];
    for (const word of words) {
      if (lines.length && (lines.at(-1)+" "+word).length <= length) lines[lines.length-1]+=" "+word;
      else lines.push(word.length>length?word.slice(0,length-1)+"…":word);
    }
    return lines.slice(0,3);
  }
  function drawGraph() {
    const result=output(), steps=result.trace.steps, shown=steps.slice(0,state.step);
    const accepted=new Set(shown.filter(s=>s.accepted).flatMap(s=>s.selected));
    const latest=new Set(shown.at(-1)?.accepted ? shown.at(-1).selected : []);
    const visible=result.edges.filter(e=>accepted.has(e.edge_id));
    const nodes=new Set([...result.anchors,...visible.flatMap(e=>[e.head,e.tail])]);
    const columns=[result.anchors.filter(n=>n!==data.target),[...nodes].filter(n=>n!==data.target&&!result.anchors.includes(n))];
    if(nodes.has(data.target))columns.push([data.target]);
    const width=$("graph").clientWidth||600;
    const height=Math.max(355,Math.max(...columns.map(c=>c.length))*120+100);
    const positions=new Map();
    const inset = width<450 ? 62 : 77;
    columns.forEach((col,i)=>col.forEach((n,j)=>positions.set(n,{x:inset+(width-2*inset)*i/Math.max(1,columns.length-1),y:45+(height-140)*(j+.5)/col.length,type:i})));
    const colors=["#377c94","#16775b","#c7705e"];
    let html=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Evidence graph at step ${state.step}"><defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#7d9f8b"/></marker></defs>`;
    visible.forEach((e,i)=>{
      const a=positions.get(e.head),b=positions.get(e.tail); if(!a||!b) return;
      const angle=Math.atan2(b.y-a.y,b.x-a.x),x1=a.x+22*Math.cos(angle),y1=a.y+22*Math.sin(angle),x2=b.x-25*Math.cos(angle),y2=b.y-25*Math.sin(angle);
      const path=a.type===b.type ? `M${a.x+22},${a.y} Q${a.x+70},${(a.y+b.y)/2} ${b.x+25},${b.y}` : `M${x1},${y1} Q${(x1+x2)/2},${(y1+y2)/2-10} ${x2},${y2}`;
      html+=`<path class="graph-edge accepted ${latest.has(e.edge_id)?"latest":""}" d="${path}" marker-end="url(#arrow)"><title>${escape(label(e.head))} → ${escape(relation(e.relation))} → ${escape(label(e.tail))}</title></path>`;
      if(a.type!==b.type) html+=`<text class="edge-label" text-anchor="middle" x="${(x1+x2)/2}" y="${(y1+y2)/2-13-i%2*9}">${escape(relation(e.relation).slice(0,23))}</text>`;
    });
    positions.forEach((p,id)=>{
      const lines=wrap(label(id),width<450?15:20);
      html+=`<g data-node="${escape(id)}"><title>${escape(label(id))} (${escape(id)})</title><circle cx="${p.x}" cy="${p.y}" r="25" fill="white" stroke="${colors[p.type]}" stroke-opacity=".18"/><circle cx="${p.x}" cy="${p.y}" r="18" fill="${colors[p.type]}" fill-opacity=".09" stroke="${colors[p.type]}" stroke-width="1.4"/><circle cx="${p.x}" cy="${p.y}" r="4" fill="${colors[p.type]}"/><text class="node-label" x="${p.x}" y="${p.y+41}">${lines.map((s,i)=>`<tspan x="${p.x}" dy="${i?13:0}">${escape(s)}</tspan>`).join("")}</text></g>`;
    });
    html+='</svg>';
    if(!steps.length) html+='<div class="empty-graph">No linked evidence · EHR fallback</div>';
    else if(!state.step) html+='<div class="empty-graph">Initial anchors</div>';
    $("graph").innerHTML=html;
    $("graph").firstElementChild.style.height=`${height}px`;
    const targetAcquired=visible.some(e=>e.head===data.target || e.tail===data.target);
    $("target-legend").hidden=!nodes.has(data.target);
    $("target-status").textContent=targetAcquired ? "Target in acquired evidence" : "No acquired target edge";
    $("target-status").title=targetAcquired ? "An accepted edge includes the prediction target concept." : "The accepted evidence contains no edge incident to the prediction target concept.";
    $("graph-status").textContent=`${accepted.size} acquired edges · ${state.step===steps.length ? (reasonLabels[result.trace.terminal_reason] || result.trace.terminal_reason) : `Step ${state.step}`}`;
  }
  document.querySelectorAll("[data-horizon]").forEach(b=>b.onclick=()=>{
    state.horizon=Number(b.dataset.horizon);
    document.querySelectorAll("[data-horizon]").forEach(t=>{t.classList.toggle("active",t===b);t.setAttribute("aria-pressed",String(t===b));});
    switchEpisode();
  });
  $("budget").onchange=e=>{state.budget=e.target.value;switchEpisode();};
  $("previous").onclick=()=>{stop();state.step=Math.max(0,state.step-1);render();};
  $("next").onclick=()=>{stop();state.step=Math.min(output().trace.steps.length,state.step+1);render();};
  $("step-slider").oninput=e=>{stop();state.step=Number(e.target.value);render();};
  $("play").onclick=()=>{
    if(state.timer){stop();icons();return;}
    if(state.step===output().trace.steps.length)state.step=0;
    $("play").innerHTML=icon("pause");$("play").title="Pause replay";$("play").setAttribute("aria-label","Pause replay");
    render();
    state.timer=setInterval(()=>{state.step++;if(state.step>=output().trace.steps.length)stop();render();},1100);
  };
  document.querySelectorAll("[data-detail]").forEach(b=>b.onclick=()=>{state.detail=b.dataset.detail;render();});
  document.querySelectorAll("[data-view]").forEach(b=>b.onclick=()=>{stop();state.view=b.dataset.view;render();});
  $("reset").onclick=()=>{Object.assign(state,{case:0,horizon:90,budget:"auto",detail:"trace",view:"explorer"});switchEpisode();};
  $("export").onclick=()=>{
    const current=data.cases[state.case];
    const blob=new Blob([JSON.stringify({case_origin:data.case_origin,case:current.name,diagnoses:current.diagnoses,history:current.history,inputs:current.inputs,horizon:state.horizon,run:{created_at:data.summary.created_at,encoder:data.summary.encoder,baseline:data.summary.baseline,policy:data.summary.policy,seed:data.summary.seed},...output()},null,2)],{type:"application/json"});
    const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download=`bar-example-${state.horizon}d.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  const summary=data.summary;
  $("cohort-stats").innerHTML=[["Training admissions",summary.cohort.train.admissions],["Validation admissions",summary.cohort.validation.admissions],["Test admissions",summary.cohort.test.admissions],["Evidence edges",summary.graph.edges]].map(([name,value])=>`<div><span>${name}</span><strong>${number(value)}</strong></div>`).join("");
  const models=[["ehr","EHR baseline"],["bar","BAR · offline policy"]];
  $("metric-table").innerHTML=summary.metrics.flatMap(row=>models.map(([id,name],i)=>`<tr class="${i===0?"group-first":""} ${id==="bar"?"bar-row":""}"><td>${i===0?row.horizon+" days":""}</td><td>${i===0?number(row.samples):""}</td><td>${i===0?number(row.positives)+" / "+percent(row.prevalence):""}</td><td>${name}</td><td>${percent(row[id].auroc)}</td><td>${percent(row[id].ap)}</td></tr>`)).join("");
  const dl=(values)=>values.map(([k,v])=>`<div><dt>${escape(k)}</dt><dd>${escape(v)}</dd></div>`).join("");
  const macro=(model,metric)=>{
    const values=summary.metrics.map(row=>row[model][metric]);
    return values.length && values.every(Number.isFinite) ? values.reduce((sum,value)=>sum+value,0)/values.length : null;
  };
  $("macro-auroc").textContent=percent(macro("bar","auroc"));
  $("macro-ap").textContent=percent(macro("bar","ap"));
  $("macro-table").innerHTML=models.map(([id,name])=>`<tr class="${id==="bar"?"bar-row":""}"><td colspan="3">CKD macro · 3 horizons</td><td>${name}</td><td>${percent(macro(id,"auroc"))}</td><td>${percent(macro(id,"ap"))}</td></tr>`).join("");
  $("reasoning-stats").innerHTML=dl([["Mean assigned budget",summary.reasoning.mean_budget.toFixed(2)],["Mean acquisition cost",summary.reasoning.mean_cost.toFixed(2)],["Fallback rate",percent(summary.reasoning.fallback_rate)],["Budget violations",summary.reasoning.budget_violations]]);
  $("run-config").innerHTML=dl([["Encoder",summary.encoder],["Baseline",summary.baseline?.name || "Stage-1 EHR-only MLP"],["Policy",summary.policy],["Split",summary.split],["Selection",summary.selection || "Fixed training schedule"],["Knowledge",summary.graph.source],["Support threshold",summary.graph.support_threshold],["Validation preference",summary.validation_preferred==="bar"?"BAR":"EHR baseline"],["Training seed",summary.seed]]);
  new ResizeObserver(()=>{if(!$("explorer-view").hidden)drawGraph();}).observe($("main-panel")||document.querySelector(".main-panel"));
  const savedStep=Number(saved.get("step"));
  state.step=saved.has("step") && Number.isInteger(savedStep) && savedStep>=0 ? Math.min(savedStep,output().trace.steps.length) : output().trace.steps.length;
  render();
})();
