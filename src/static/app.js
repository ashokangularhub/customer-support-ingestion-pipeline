const state = {
  collection: null,
  offset: 0,
  limit: 25,
  total: 0,
  filters: { doc_type: "", content_type: "", source_file: "", search: "" },
  includeVector: false,
};

const el = (id) => document.getElementById(id);

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

function truncate(text, n = 140) {
  if (!text) return "";
  return text.length > n ? text.slice(0, n) + "…" : text;
}

async function loadCollections() {
  const collections = await fetchJSON("/api/collections");
  const select = el("collection-select");
  select.innerHTML = "";
  collections.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c.name;
    opt.textContent = `${c.name} (${c.count})`;
    select.appendChild(opt);
  });
  if (collections.length) {
    state.collection = collections[0].name;
    el("collection-count").textContent = `${collections[0].count} objects`;
  }
}

async function loadFacets() {
  if (!state.collection) return;
  const facets = await fetchJSON(`/api/collections/${state.collection}/facets`);
  fillFacetSelect("filter-doc-type", facets.doc_type || []);
  fillFacetSelect("filter-content-type", facets.content_type || []);
  fillFacetSelect("filter-source-file", facets.source_file || []);
}

function fillFacetSelect(id, values) {
  const select = el(id);
  const current = select.value;
  select.innerHTML = '<option value="">All</option>';
  values.forEach((v) => {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    select.appendChild(opt);
  });
  select.value = current;
}

function buildQuery() {
  const params = new URLSearchParams({
    limit: state.limit,
    offset: state.offset,
    include_vector: state.includeVector,
  });
  if (state.filters.doc_type) params.set("doc_type", state.filters.doc_type);
  if (state.filters.content_type) params.set("content_type", state.filters.content_type);
  if (state.filters.source_file) params.set("source_file", state.filters.source_file);
  if (state.filters.search) params.set("search", state.filters.search);
  return params.toString();
}

async function loadObjects() {
  if (!state.collection) return;
  const data = await fetchJSON(
    `/api/collections/${state.collection}/objects?${buildQuery()}`
  );
  state.total = data.total;
  renderTable(data.objects);
  renderPager();
}

function renderTable(objects) {
  const body = el("results-body");
  body.innerHTML = "";
  el("empty-state").classList.toggle("hidden", objects.length > 0);

  objects.forEach((obj) => {
    const p = obj.properties;
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${p.page_number ?? ""}</td>
      <td><span class="tag ${p.content_type || ""}">${p.content_type || ""}</span></td>
      <td>${p.doc_type || ""}</td>
      <td>${p.table_name || p.section_heading || ""}</td>
      <td>${truncate(p.text)}</td>
      <td>${p.product_name || ""}</td>
    `;
    row.addEventListener("click", () => showDetail(obj));
    body.appendChild(row);
  });
}

function renderPager() {
  const page = Math.floor(state.offset / state.limit) + 1;
  const pageCount = Math.max(1, Math.ceil(state.total / state.limit));
  el("page-indicator").textContent = `Page ${page} of ${pageCount} (${state.total} total)`;
  el("prev-page").disabled = state.offset === 0;
  el("next-page").disabled = state.offset + state.limit >= state.total;
}

function showDetail(obj) {
  const p = obj.properties;
  const rows = Object.entries(p)
    .map(([k, v]) => `<tr><td>${k}</td><td>${formatValue(v)}</td></tr>`)
    .join("");

  let vectorHtml = "";
  if (obj.vector_preview) {
    vectorHtml = `<p><strong>Vector</strong> (dim=${obj.vector_dim}): first 8 dims</p>
      <pre>${JSON.stringify(obj.vector_preview)}</pre>
      <button id="load-full-vector">Load full vector</button>
      <div id="full-vector"></div>`;
  } else {
    vectorHtml = `<button id="load-full-vector">Load vector</button><div id="full-vector"></div>`;
  }

  el("detail-content").innerHTML = `
    <p class="muted">UUID: ${obj.uuid}</p>
    <table class="kv-table">${rows}</table>
    ${vectorHtml}
  `;
  el("detail-overlay").classList.remove("hidden");

  const loadBtn = el("load-full-vector");
  if (loadBtn) {
    loadBtn.addEventListener("click", async () => {
      const full = await fetchJSON(`/api/collections/${state.collection}/objects/${obj.uuid}`);
      el("full-vector").innerHTML = `<p>Full vector (dim=${full.vector_dim}):</p><pre>${JSON.stringify(full.vector)}</pre>`;
    });
  }
}

function formatValue(v) {
  if (v === null || v === undefined) return "";
  if (typeof v === "string" && v.length > 300) return truncate(v, 300);
  return String(v);
}

function resetToFirstPage() {
  state.offset = 0;
  loadObjects();
}

function wireEvents() {
  el("collection-select").addEventListener("change", async (e) => {
    state.collection = e.target.value;
    state.offset = 0;
    const selected = e.target.selectedOptions[0];
    el("collection-count").textContent = selected ? selected.textContent : "";
    await loadFacets();
    await loadObjects();
  });

  el("filter-doc-type").addEventListener("change", (e) => {
    state.filters.doc_type = e.target.value;
    resetToFirstPage();
  });
  el("filter-content-type").addEventListener("change", (e) => {
    state.filters.content_type = e.target.value;
    resetToFirstPage();
  });
  el("filter-source-file").addEventListener("change", (e) => {
    state.filters.source_file = e.target.value;
    resetToFirstPage();
  });
  el("filter-search").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      state.filters.search = e.target.value.trim();
      resetToFirstPage();
    }
  });
  el("page-size").addEventListener("change", (e) => {
    state.limit = Number(e.target.value);
    resetToFirstPage();
  });
  el("include-vector").addEventListener("change", (e) => {
    state.includeVector = e.target.checked;
    loadObjects();
  });

  el("prev-page").addEventListener("click", () => {
    state.offset = Math.max(0, state.offset - state.limit);
    loadObjects();
  });
  el("next-page").addEventListener("click", () => {
    state.offset += state.limit;
    loadObjects();
  });

  el("close-detail").addEventListener("click", () => {
    el("detail-overlay").classList.add("hidden");
  });
  el("detail-overlay").addEventListener("click", (e) => {
    if (e.target.id === "detail-overlay") el("detail-overlay").classList.add("hidden");
  });
}

(async function init() {
  wireEvents();
  await loadCollections();
  await loadFacets();
  await loadObjects();
})();
