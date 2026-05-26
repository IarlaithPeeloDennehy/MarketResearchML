function esc(s){const d=document.createElement('div');d.textContent=String(s??'');return d.innerHTML;}

/* ================================================================
   BACKEND CONFIG
================================================================ */
// Empty string = same-origin relative paths (/analyse, /backtest, etc.)
// The frontend is now served by FastAPI, so all fetch() calls are same-origin.
// For local dev: run the backend and open http://localhost:8000 — not file://
const BACKEND_URL = '';

async function callBackend(endpoint, payload) {
  try {
    const r = await fetch(BACKEND_URL + endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify(payload),
    });
    if (!r.ok) { console.warn('Backend HTTP ' + r.status); return {ok:false, status:r.status, data:null}; }
    return {ok:true, status:r.status, data: await r.json()};
  } catch(e) {
    console.warn('Backend unreachable:', e.message);
    return {ok:false, status:0, data:null};
  }
}

async function checkBackend() {
  try { const r = await fetch(BACKEND_URL + '/health', { credentials: 'include' }); return r.ok; }
  catch { return false; }
}

/* ═══ MOBILE SIDEBAR TOGGLE ═══ */
const hamburger = document.getElementById('hamburger');
const sidebar = document.getElementById('sidebar');
const backdrop = document.getElementById('sidebarBackdrop');

function openSidebar(){
  sidebar.classList.add('open');
  backdrop.classList.add('visible');
  hamburger.classList.add('open');
  document.body.style.overflow='hidden';
}
function closeSidebar(){
  sidebar.classList.remove('open');
  backdrop.classList.remove('visible');
  hamburger.classList.remove('open');
  document.body.style.overflow='';
}
function toggleSidebar(){
  sidebar.classList.contains('open') ? closeSidebar() : openSidebar();
}
hamburger.addEventListener('click', toggleSidebar);
backdrop.addEventListener('click', closeSidebar);

function runAnalysisAndClose(){
  closeSidebar();
  runAnalysis();
}

/* ═══ PAGE NAVIGATION ═══ */
let spActiveMkt='US';
let spRiskKey='medium';

function showPage(name){
  document.getElementById('welcomeScreen').style.display=(name==='landing')?'':'none';
  document.getElementById('selectScreen').style.display=(name==='select')?'block':'none';
  document.getElementById('resultsScreen').style.display=(name==='results')?'block':'none';
  document.querySelector('main.main').style.display=(name==='select')?'none':'';
  const sp=document.getElementById('savedPanel');
  if(sp)sp.style.display=(name==='landing')?'':'none';
  if(name==='select'){renderStockGrid('');renderSelectedList();}
}

/* ═══ SEARCH UTILITIES ═══ */

// Client-side TTL cache so repeated keystrokes never hit the server twice.
// Mirrors the server-side cache — results are stable within a session.
const SearchCache=(()=>{
  const TTL=2*60*1000; // 2 min
  const MAX=100;
  const store=new Map();
  return{
    get(key){
      const e=store.get(key);
      if(!e)return null;
      if(Date.now()-e.ts>TTL){store.delete(key);return null;}
      return e.results;
    },
    set(key,results){
      if(store.size>=MAX)store.delete(store.keys().next().value);
      store.set(key,{ts:Date.now(),results});
    },
  };
})();

// Recent searches persisted in localStorage across page loads.
const RecentSearches=(()=>{
  const KEY='numkt_recent';
  const MAX=8;
  const load=()=>{try{return JSON.parse(localStorage.getItem(KEY)||'[]');}catch{return[];}};
  return{
    get(){return load();},
    add(q){if(!q||q.length<2)return;const a=load().filter(x=>x!==q);a.unshift(q);try{localStorage.setItem(KEY,JSON.stringify(a.slice(0,MAX)));}catch{}},
    clear(){try{localStorage.removeItem(KEY);}catch{}},
  };
})();

// Wrap a query fragment with <mark> for the matched portion.
// Uses esc() to prevent XSS on both halves before injecting the tag.
function highlightMatch(text,query){
  if(!query)return esc(text);
  const idx=text.toLowerCase().indexOf(query.toLowerCase());
  if(idx===-1)return esc(text);
  return esc(text.slice(0,idx))+'<mark class="search-hl">'+esc(text.slice(idx,idx+query.length))+'</mark>'+esc(text.slice(idx+query.length));
}

// Hardcoded popular tickers shown as a starting point when the input is focused
// but empty — these are always in ALL_STOCKS so the Add button works immediately.
const POPULAR_TICKERS=[
  {t:'AAPL',n:'Apple Inc.'},{t:'MSFT',n:'Microsoft Corp.'},
  {t:'NVDA',n:'NVIDIA Corp.'},{t:'GOOGL',n:'Alphabet Inc.'},
  {t:'AMZN',n:'Amazon.com Inc.'},{t:'META',n:'Meta Platforms'},
  {t:'TSLA',n:'Tesla Inc.'},{t:'JPM',n:'JPMorgan Chase'},
];

/* ═══ SELECT PAGE — STOCK GRID ═══ */
function renderStockGrid(query){
  const grid=document.getElementById('sp-stockGrid');
  if(!grid)return;
  let pool=spActiveMkt==='ALL'?ALL_STOCKS:ALL_STOCKS.filter(s=>s.m===spActiveMkt);
  if(query){
    const q=query.toLowerCase();
    pool=pool.filter(s=>s.t.toLowerCase().includes(q)||s.n.toLowerCase().includes(q));
  }
  if(!pool.length){
    grid.innerHTML='<div style="grid-column:1/-1;text-align:center;padding:40px 0;font-family:\'JetBrains Mono\',monospace;font-size:11px;color:var(--text3)">No stocks match — try searching below.</div>';
  } else {
    grid.innerHTML=pool.map(s=>{
      const mc=s.m==='US'?'mkt-us':s.m==='UK'?'mkt-uk':'mkt-ie';
      const sel=selectedTickers.has(s.t);
      return`<div class="stock-card${sel?' selected':''}" data-tk="${esc(s.t)}" onclick="toggleStock('${esc(s.t)}')">
        <div class="sc-check"><svg class="sc-check-mark" viewBox="0 0 8 6" fill="none"><path d="M1 3L3 5L7 1" stroke="#0a0b14" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
        <div class="sc-ticker">${esc(s.t)}</div>
        <div class="sc-name">${esc(s.n)}</div>
        <div class="sc-badges"><span class="sr-mkt ${mc}">${s.m}</span><span class="sc-sector">${esc(s.s)}</span></div>
      </div>`;
    }).join('');
  }
  if(query&&query.length>=1)_debouncedApiSearch(query);
  else{
    const rs=document.getElementById('sp-searchResults');
    if(rs)rs.style.display='none';
  }
}

/* ═══ SELECT PAGE — FINNHUB LIVE SEARCH ═══ */

// Called after the local ALL_STOCKS grid is filtered. Shows additional tickers
// from Finnhub that are NOT already in the curated list.
const _debouncedApiSearch=debounce(async function(query){
  const sec=document.getElementById('sp-searchResults');
  const lst=document.getElementById('sp-searchResultsList');
  if(!sec||!lst)return;

  // Client-side cache check — avoids a round-trip for identical queries
  const cacheKey=query.toLowerCase();
  const cached=SearchCache.get(cacheKey);
  if(cached){_renderApiResults(lst,cached,query,sec);return;}

  sec.style.display='';
  lst.innerHTML='<div class="sp-search-spinner">Searching Finnhub…</div>';

  try{
    const r=await fetch(`/api/search?q=${encodeURIComponent(query)}&types=stock`);
    if(!r.ok)throw new Error(`HTTP ${r.status}`);
    const d=await r.json();
    // Filter out tickers already in the curated ALL_STOCKS grid
    const hits=(d.results||[]).filter(x=>!ALL_STOCKS.find(s=>s.t===x.ticker));
    SearchCache.set(cacheKey,hits);
    if(hits.length)RecentSearches.add(query);
    _renderApiResults(lst,hits,query,sec);
  }catch(e){
    lst.innerHTML='<div class="sp-search-spinner" style="color:var(--red)">Search unavailable — try again.</div>';
  }
},350);

function _renderApiResults(lst,hits,query,sec){
  if(!hits.length){if(sec)sec.style.display='none';return;}
  if(sec)sec.style.display='';
  lst.innerHTML=hits.map(r=>{
    const isSel=selectedTickers.has(r.ticker);
    const exBadge=r.exchange?`<span class="sp-sr-exch">${esc(r.exchange)}</span>`:'';
    return`<div class="sp-search-result-row" tabindex="0">
      <span class="sp-sr-ticker">${highlightMatch(r.ticker,query)}</span>
      <span class="sp-sr-name">${highlightMatch(r.name,query)}</span>
      ${exBadge}
      <button class="sp-sr-add${isSel?' selected':''}" ${isSel?'disabled':''}
        data-ticker="${esc(r.ticker)}" data-name="${esc(r.name)}"
        onclick="addCustomStock(this.dataset.ticker,this.dataset.name)">
        ${isSel?'Added':'+ Add'}
      </button>
    </div>`;
  }).join('');
}

function renderSelectedList(){
  const list=document.getElementById('sp-selectedList');
  const countEl=document.getElementById('sp-count');
  const btn=document.getElementById('sp-runBtn');
  const hint=document.getElementById('sp-runHint');
  if(!list)return;
  const arr=[...selectedTickers];
  list.innerHTML=arr.length
    ?arr.map(tk=>{
      const isCustom=customStocks.has(tk)&&!ALL_STOCKS.find(s=>s.t===tk);
      return`<div class="sp-chip${isCustom?' sp-chip-custom':''}">${esc(tk)}<button onclick="toggleStock('${esc(tk)}')" title="Remove">×</button></div>`;
    }).join('')
    :'<span style="font-size:11px;color:var(--text3)">No stocks selected yet.</span>';
  if(countEl)countEl.textContent=`${arr.length} / 40`;
  if(btn)btn.disabled=arr.length<3;
  if(hint)hint.style.display=arr.length<3?'block':'none';
}

function toggleStock(tk){
  if(selectedTickers.has(tk)){
    selectedTickers.delete(tk);
  } else if(selectedTickers.size<40){
    selectedTickers.add(tk);
  }
  const card=document.querySelector(`#sp-stockGrid .stock-card[data-tk="${CSS.escape(tk)}"]`);
  if(card){card.classList.toggle('selected',selectedTickers.has(tk));}
  renderSelectedList();
  initStockList();
}

/* ═══ SELECT PAGE EVENT LISTENERS ═══ */
document.getElementById('sp-searchInput').addEventListener('input',e=>renderStockGrid(e.target.value.trim()));

// Show popular/recent suggestions when the input is focused but empty.
document.getElementById('sp-searchInput').addEventListener('focus',e=>{
  if(!e.target.value.trim())_showSearchSuggestions();
});

// Delay hide so click events on result rows have time to fire first.
document.getElementById('sp-searchInput').addEventListener('blur',()=>{
  setTimeout(()=>{
    const sec=document.getElementById('sp-searchResults');
    if(sec&&!sec.querySelector(':focus-within'))sec.style.display='none';
  },200);
});

// ── Keyboard navigation for the Finnhub results list ──
document.getElementById('sp-searchInput').addEventListener('keydown',e=>{
  const lst=document.getElementById('sp-searchResultsList');
  if(!lst)return;
  const rows=[...lst.querySelectorAll('.sp-search-result-row')];
  const active=lst.querySelector('.sp-search-result-row.kb-active');
  const idx=active?rows.indexOf(active):-1;

  if(e.key==='ArrowDown'){
    e.preventDefault();
    if(!rows.length)return;
    const next=rows[(idx+1)%rows.length];
    active&&active.classList.remove('kb-active');
    next.classList.add('kb-active');
    next.scrollIntoView({block:'nearest'});
  } else if(e.key==='ArrowUp'){
    e.preventDefault();
    if(!rows.length)return;
    const prev=rows[(idx<=0?rows.length:idx)-1];
    active&&active.classList.remove('kb-active');
    prev.classList.add('kb-active');
    prev.scrollIntoView({block:'nearest'});
  } else if(e.key==='Enter'&&active){
    e.preventDefault();
    const btn=active.querySelector('.sp-sr-add:not(:disabled)');
    if(btn)btn.click();
  } else if(e.key==='Escape'){
    const sec=document.getElementById('sp-searchResults');
    if(sec)sec.style.display='none';
    e.target.blur();
  }
});

// Close search results when clicking outside the search area.
document.addEventListener('click',e=>{
  if(!e.target.closest('.sp-search-wrap')&&!e.target.closest('.sp-search-results')){
    const sec=document.getElementById('sp-searchResults');
    if(sec)sec.style.display='none';
  }
});

function _showSearchSuggestions(){
  const sec=document.getElementById('sp-searchResults');
  const lst=document.getElementById('sp-searchResultsList');
  if(!sec||!lst)return;

  const recent=RecentSearches.get();
  let html='';

  if(recent.length){
    html+=`<div class="sp-search-results-label" style="display:flex;justify-content:space-between;align-items:center">
      Recent <button class="sp-sr-clear" onclick="RecentSearches.clear();_showSearchSuggestions()">Clear</button>
    </div>`;
    html+=recent.map(q=>`<div class="sp-search-result-row sp-sr-recent" onclick="document.getElementById('sp-searchInput').value='${esc(q)}';renderStockGrid('${esc(q)}')">
      <span class="sp-sr-ticker" style="font-size:11px;font-weight:400">${esc(q)}</span>
      <span class="sp-sr-name" style="color:var(--text3)">recent search</span>
    </div>`).join('');
    html+='<div class="sp-search-results-label" style="margin-top:10px">Popular</div>';
  } else {
    html+='<div class="sp-search-results-label">Popular Tickers</div>';
  }

  html+=POPULAR_TICKERS.map(p=>{
    const isSel=selectedTickers.has(p.t);
    return`<div class="sp-search-result-row">
      <span class="sp-sr-ticker">${esc(p.t)}</span>
      <span class="sp-sr-name">${esc(p.n)}</span>
      <button class="sp-sr-add${isSel?' selected':''}" ${isSel?'disabled':''}
        data-ticker="${esc(p.t)}" data-name="${esc(p.n)}"
        onclick="addCustomStock(this.dataset.ticker,this.dataset.name)">
        ${isSel?'Added':'+ Add'}
      </button>
    </div>`;
  }).join('');

  lst.innerHTML=html;
  sec.style.display='';
}

document.getElementById('sp-lbSlider').addEventListener('input',function(){
  document.getElementById('sp-lbVal').textContent=this.value+' yr';
});

document.querySelectorAll('#sp-marketTabs .sp-market-tab').forEach(tab=>{
  tab.onclick=()=>{
    spActiveMkt=tab.dataset.mkt;
    document.querySelectorAll('#sp-marketTabs .sp-market-tab').forEach(t=>t.classList.remove('active'));
    tab.classList.add('active');
    renderStockGrid(document.getElementById('sp-searchInput').value.trim());
  };
});

document.querySelectorAll('#sp-riskTabs .risk-tab').forEach(tab=>{
  tab.onclick=()=>{
    spRiskKey=tab.dataset.risk;
    document.querySelectorAll('#sp-riskTabs .risk-tab').forEach(t=>t.classList.remove('active'));
    tab.classList.add('active');
  };
});

/* ═══ STOCK UNIVERSE ═══ */
const ALL_STOCKS=[
{t:'AAPL', n:'Apple Inc.',            s:'Technology',      m:'US',p:189.5, pe:29, pb:45,  roe:0.87,de:1.72,dy:0.005,rg:0.08, mg:0.254,beta:1.20,m12:0.22, m3:0.04, mc:3000},
{t:'MSFT', n:'Microsoft Corp.',       s:'Technology',      m:'US',p:415.2, pe:36, pb:14,  roe:0.43,de:0.35,dy:0.007,rg:0.16, mg:0.375,beta:0.90,m12:0.32, m3:0.08, mc:3100},
{t:'GOOGL',n:'Alphabet Inc.',         s:'Technology',      m:'US',p:175.0, pe:24, pb:6.2, roe:0.28,de:0.10,dy:0.000,rg:0.14, mg:0.265,beta:1.10,m12:0.36, m3:0.09, mc:2200},
{t:'AMZN', n:'Amazon.com Inc.',       s:'Consumer Disc.',  m:'US',p:182.0, pe:55, pb:8.5, roe:0.16,de:0.55,dy:0.000,rg:0.12, mg:0.060,beta:1.35,m12:0.44, m3:0.12, mc:1900},
{t:'NVDA', n:'NVIDIA Corp.',          s:'Technology',      m:'US',p:880.0, pe:68, pb:38,  roe:0.55,de:0.42,dy:0.001,rg:0.85, mg:0.536,beta:1.75,m12:2.10, m3:0.45, mc:2160},
{t:'META', n:'Meta Platforms',        s:'Technology',      m:'US',p:485.0, pe:23, pb:6.8, roe:0.32,de:0.12,dy:0.004,rg:0.22, mg:0.406,beta:1.40,m12:0.75, m3:0.18, mc:1250},
{t:'BRK-B',n:'Berkshire Hathaway',    s:'Financials',      m:'US',p:358.0, pe:9,  pb:1.5, roe:0.16,de:0.25,dy:0.000,rg:0.12, mg:0.090,beta:0.80,m12:0.18, m3:0.06, mc:870},
{t:'JPM',  n:'JPMorgan Chase',        s:'Financials',      m:'US',p:198.4, pe:11, pb:2.0, roe:0.17,de:1.15,dy:0.024,rg:0.21, mg:0.320,beta:1.10,m12:0.28, m3:0.07, mc:570},
{t:'JNJ',  n:'Johnson & Johnson',     s:'Healthcare',      m:'US',p:152.3, pe:16, pb:5.5, roe:0.22,de:0.48,dy:0.030,rg:0.03, mg:0.190,beta:0.60,m12:-0.04,m3:-0.02,mc:365},
{t:'UNH',  n:'UnitedHealth Group',    s:'Healthcare',      m:'US',p:492.0, pe:20, pb:5.2, roe:0.27,de:0.70,dy:0.015,rg:0.09, mg:0.073,beta:0.65,m12:0.08, m3:-0.06,mc:453},
{t:'XOM',  n:'ExxonMobil',            s:'Energy',          m:'US',p:112.0, pe:14, pb:2.1, roe:0.18,de:0.20,dy:0.035,rg:0.05, mg:0.098,beta:1.00,m12:0.12, m3:0.03, mc:448},
{t:'CVX',  n:'Chevron Corp.',         s:'Energy',          m:'US',p:155.0, pe:13, pb:1.8, roe:0.14,de:0.16,dy:0.040,rg:-0.02,mg:0.095,beta:0.95,m12:0.05, m3:0.02, mc:286},
{t:'PG',   n:'Procter & Gamble',      s:'Consumer Staples',m:'US',p:165.8, pe:23, pb:7.8, roe:0.34,de:0.62,dy:0.024,rg:0.04, mg:0.195,beta:0.55,m12:0.06, m3:0.01, mc:393},
{t:'WMT',  n:'Walmart Inc.',          s:'Consumer Staples',m:'US',p:68.0,  pe:28, pb:7.5, roe:0.27,de:0.72,dy:0.013,rg:0.06, mg:0.028,beta:0.50,m12:0.32, m3:0.07, mc:547},
{t:'COST', n:'Costco Wholesale',      s:'Consumer Staples',m:'US',p:720.0, pe:48, pb:13,  roe:0.28,de:0.42,dy:0.006,rg:0.06, mg:0.028,beta:0.72,m12:0.28, m3:0.09, mc:319},
{t:'KO',   n:'Coca-Cola Co.',         s:'Consumer Staples',m:'US',p:62.5,  pe:24, pb:10,  roe:0.42,de:1.95,dy:0.030,rg:0.03, mg:0.225,beta:0.58,m12:0.08, m3:0.02, mc:270},
{t:'PEP',  n:'PepsiCo Inc.',          s:'Consumer Staples',m:'US',p:168.0, pe:22, pb:12,  roe:0.55,de:2.80,dy:0.030,rg:0.03, mg:0.148,beta:0.55,m12:0.02, m3:-0.01,mc:231},
{t:'V',    n:'Visa Inc.',             s:'Financials',      m:'US',p:278.0, pe:30, pb:14,  roe:0.46,de:0.60,dy:0.008,rg:0.10, mg:0.535,beta:0.95,m12:0.18, m3:0.05, mc:570},
{t:'MA',   n:'Mastercard Inc.',       s:'Financials',      m:'US',p:480.0, pe:34, pb:58,  roe:1.80,de:2.30,dy:0.006,rg:0.12, mg:0.465,beta:1.00,m12:0.22, m3:0.06, mc:452},
{t:'BAC',  n:'Bank of America',       s:'Financials',      m:'US',p:38.5,  pe:12, pb:1.3, roe:0.11,de:1.05,dy:0.026,rg:0.08, mg:0.270,beta:1.30,m12:0.24, m3:0.08, mc:305},
{t:'WFC',  n:'Wells Fargo',           s:'Financials',      m:'US',p:55.0,  pe:11, pb:1.3, roe:0.12,de:1.00,dy:0.025,rg:0.06, mg:0.265,beta:1.20,m12:0.38, m3:0.10, mc:205},
{t:'GS',   n:'Goldman Sachs',         s:'Financials',      m:'US',p:465.0, pe:13, pb:1.5, roe:0.12,de:3.20,dy:0.026,rg:0.16, mg:0.195,beta:1.35,m12:0.32, m3:0.09, mc:158},
{t:'TSLA', n:'Tesla Inc.',            s:'Consumer Disc.',  m:'US',p:175.0, pe:55, pb:9.0, roe:0.17,de:0.15,dy:0.000,rg:0.19, mg:0.073,beta:2.10,m12:-0.28,m3:-0.12,mc:558},
{t:'NFLX', n:'Netflix Inc.',          s:'Technology',      m:'US',p:625.0, pe:42, pb:14,  roe:0.32,de:1.10,dy:0.000,rg:0.15, mg:0.165,beta:1.20,m12:0.55, m3:0.14, mc:270},
{t:'ADBE', n:'Adobe Inc.',            s:'Technology',      m:'US',p:475.0, pe:35, pb:14,  roe:0.40,de:0.45,dy:0.000,rg:0.11, mg:0.310,beta:1.15,m12:0.10, m3:0.02, mc:214},
{t:'CRM',  n:'Salesforce Inc.',       s:'Technology',      m:'US',p:298.0, pe:42, pb:4.5, roe:0.11,de:0.18,dy:0.000,rg:0.11, mg:0.155,beta:1.25,m12:0.22, m3:0.06, mc:290},
{t:'ORCL', n:'Oracle Corp.',          s:'Technology',      m:'US',p:118.0, pe:22, pb:99,  roe:2.80,de:7.20,dy:0.012,rg:0.08, mg:0.215,beta:0.90,m12:0.34, m3:0.09, mc:326},
{t:'AMD',  n:'Advanced Micro Devices',s:'Technology',      m:'US',p:160.0, pe:45, pb:3.8, roe:0.09,de:0.08,dy:0.000,rg:0.06, mg:0.048,beta:1.80,m12:0.28, m3:0.07, mc:258},
{t:'INTC', n:'Intel Corp.',           s:'Technology',      m:'US',p:22.0,  pe:99, pb:0.9, roe:-0.02,de:0.50,dy:0.020,rg:-0.14,mg:-0.028,beta:1.00,m12:-0.42,m3:-0.15,mc:93},
{t:'QCOM', n:'Qualcomm Inc.',         s:'Technology',      m:'US',p:158.0, pe:16, pb:6.5, roe:0.42,de:1.50,dy:0.025,rg:0.12, mg:0.265,beta:1.20,m12:0.18, m3:0.04, mc:172},
{t:'AVGO', n:'Broadcom Inc.',         s:'Technology',      m:'US',p:1320.0,pe:28, pb:8.0, roe:0.30,de:1.80,dy:0.015,rg:0.12, mg:0.245,beta:1.10,m12:0.42, m3:0.11, mc:618},
{t:'LLY',  n:'Eli Lilly',            s:'Healthcare',      m:'US',p:820.0, pe:62, pb:58,  roe:0.90,de:2.20,dy:0.006,rg:0.52, mg:0.335,beta:0.55,m12:0.92, m3:0.22, mc:780},
{t:'ABBV', n:'AbbVie Inc.',           s:'Healthcare',      m:'US',p:168.0, pe:18, pb:60,  roe:3.20,de:3.80,dy:0.038,rg:0.14, mg:0.265,beta:0.65,m12:0.22, m3:0.06, mc:297},
{t:'MRK',  n:'Merck & Co.',           s:'Healthcare',      m:'US',p:128.0, pe:14, pb:5.8, roe:0.42,de:0.85,dy:0.025,rg:0.14, mg:0.295,beta:0.50,m12:0.14, m3:0.03, mc:325},
{t:'PFE',  n:'Pfizer Inc.',           s:'Healthcare',      m:'US',p:27.0,  pe:10, pb:1.5, roe:0.06,de:0.60,dy:0.065,rg:-0.42,mg:0.045,beta:0.55,m12:-0.32,m3:-0.08,mc:152},
{t:'NEE',  n:'NextEra Energy',        s:'Utilities',       m:'US',p:72.0,  pe:20, pb:2.8, roe:0.14,de:1.40,dy:0.030,rg:0.08, mg:0.195,beta:0.60,m12:0.08, m3:0.02, mc:147},
{t:'DUK',  n:'Duke Energy',           s:'Utilities',       m:'US',p:98.0,  pe:18, pb:1.8, roe:0.10,de:1.60,dy:0.042,rg:0.04, mg:0.185,beta:0.45,m12:0.06, m3:0.01, mc:75},
{t:'CAT',  n:'Caterpillar Inc.',      s:'Industrials',     m:'US',p:328.0, pe:16, pb:10,  roe:0.62,de:2.00,dy:0.016,rg:0.06, mg:0.175,beta:1.10,m12:0.24, m3:0.06, mc:163},
{t:'HON',  n:'Honeywell Intl.',       s:'Industrials',     m:'US',p:195.0, pe:22, pb:7.5, roe:0.34,de:1.10,dy:0.022,rg:0.05, mg:0.155,beta:0.90,m12:0.06, m3:0.01, mc:129},
{t:'RTX',  n:'RTX Corp.',             s:'Industrials',     m:'US',p:118.0, pe:32, pb:2.1, roe:0.07,de:0.60,dy:0.022,rg:0.10, mg:0.086,beta:0.85,m12:0.14, m3:0.04, mc:155},
{t:'SHEL.L', n:'Shell plc',           s:'Energy',          m:'UK',p:28.5,  pe:12, pb:1.3, roe:0.13,de:0.35,dy:0.040,rg:0.03, mg:0.095,beta:0.90,m12:0.08, m3:0.02, mc:218},
{t:'AZN.L',  n:'AstraZeneca plc',     s:'Healthcare',      m:'UK',p:121.5, pe:30, pb:6.5, roe:0.22,de:0.85,dy:0.020,rg:0.18, mg:0.310,beta:0.65,m12:0.14, m3:0.04, mc:192},
{t:'HSBA.L', n:'HSBC Holdings',       s:'Financials',      m:'UK',p:7.20,  pe:8,  pb:0.9, roe:0.12,de:0.95,dy:0.065,rg:0.12, mg:0.280,beta:0.75,m12:0.22, m3:0.06, mc:148},
{t:'ULVR.L', n:'Unilever plc',        s:'Consumer Staples',m:'UK',p:42.0,  pe:19, pb:5.0, roe:0.28,de:0.95,dy:0.037,rg:0.02, mg:0.155,beta:0.50,m12:0.06, m3:0.01, mc:109},
{t:'BP.L',   n:'BP plc',              s:'Energy',          m:'UK',p:4.60,  pe:10, pb:1.1, roe:0.11,de:0.52,dy:0.048,rg:-0.04,mg:0.070,beta:1.00,m12:-0.08,m3:-0.03,mc:85},
{t:'GSK.L',  n:'GSK plc',             s:'Healthcare',      m:'UK',p:17.2,  pe:14, pb:3.5, roe:0.26,de:0.72,dy:0.038,rg:0.07, mg:0.285,beta:0.55,m12:0.12, m3:0.03, mc:73},
{t:'RIO.L',  n:'Rio Tinto plc',       s:'Materials',       m:'UK',p:52.0,  pe:11, pb:2.2, roe:0.21,de:0.28,dy:0.055,rg:0.04, mg:0.195,beta:0.85,m12:0.04, m3:-0.01,mc:88},
{t:'BHP.L',  n:'BHP Group',           s:'Materials',       m:'UK',p:22.5,  pe:12, pb:2.4, roe:0.22,de:0.32,dy:0.058,rg:0.02, mg:0.215,beta:0.90,m12:0.06, m3:0.01, mc:115},
{t:'BARC.L', n:'Barclays plc',        s:'Financials',      m:'UK',p:2.45,  pe:7,  pb:0.7, roe:0.10,de:1.20,dy:0.038,rg:0.09, mg:0.220,beta:1.20,m12:0.35, m3:0.09, mc:44},
{t:'DGE.L',  n:'Diageo plc',          s:'Consumer Staples',m:'UK',p:24.5,  pe:17, pb:5.5, roe:0.34,de:1.10,dy:0.035,rg:-0.01,mg:0.265,beta:0.55,m12:-0.18,m3:-0.06,mc:62},
{t:'LSEG.L', n:'LSEG plc',            s:'Financials',      m:'UK',p:100.0, pe:32, pb:3.0, roe:0.10,de:0.55,dy:0.013,rg:0.08, mg:0.190,beta:0.70,m12:0.15, m3:0.04, mc:48},
{t:'STAN.L', n:'Standard Chartered',  s:'Financials',      m:'UK',p:9.80,  pe:9,  pb:0.65,roe:0.09,de:0.85,dy:0.032,rg:0.14, mg:0.250,beta:0.90,m12:0.28, m3:0.07, mc:32},
{t:'NWG.L',  n:'NatWest Group',       s:'Financials',      m:'UK',p:3.85,  pe:8,  pb:0.95,roe:0.12,de:0.88,dy:0.055,rg:0.06, mg:0.290,beta:1.00,m12:0.38, m3:0.10, mc:36},
{t:'LLOY.L', n:'Lloyds Banking Group',s:'Financials',      m:'UK',p:0.575, pe:9,  pb:0.8, roe:0.11,de:0.92,dy:0.060,rg:0.04, mg:0.255,beta:1.05,m12:0.28, m3:0.07, mc:41},
{t:'RR.L',   n:'Rolls-Royce Holdings',s:'Industrials',     m:'UK',p:4.50,  pe:22, pb:99,  roe:0.45,de:4.50,dy:0.000,rg:0.18, mg:0.085,beta:1.40,m12:1.20, m3:0.28, mc:40},
{t:'BA.L',   n:'BAE Systems',         s:'Industrials',     m:'UK',p:13.5,  pe:20, pb:8.0, roe:0.38,de:0.80,dy:0.022,rg:0.12, mg:0.095,beta:0.55,m12:0.32, m3:0.09, mc:44},
{t:'VOD.L',  n:'Vodafone Group',      s:'Telecom',         m:'UK',p:0.755, pe:11, pb:0.55,roe:0.05,de:1.50,dy:0.115,rg:-0.03,mg:0.045,beta:0.80,m12:-0.22,m3:-0.06,mc:21},
{t:'BT.L',   n:'BT Group',            s:'Telecom',         m:'UK',p:1.62,  pe:8,  pb:1.1, roe:0.12,de:1.65,dy:0.055,rg:-0.01,mg:0.075,beta:0.85,m12:0.18, m3:0.05, mc:16},
{t:'EXPN.L', n:'Experian plc',        s:'Technology',      m:'UK',p:38.0,  pe:32, pb:10,  roe:0.32,de:1.20,dy:0.015,rg:0.08, mg:0.235,beta:0.70,m12:0.14, m3:0.04, mc:34},
{t:'AUTO.L', n:'Auto Trader Group',   s:'Technology',      m:'UK',p:7.90,  pe:27, pb:14,  roe:0.55,de:0.45,dy:0.018,rg:0.08, mg:0.535,beta:0.75,m12:0.22, m3:0.06, mc:9},
{t:'FERG.L', n:'Ferguson Enterprises',s:'Industrials',     m:'UK',p:160.0, pe:22, pb:6.5, roe:0.30,de:0.60,dy:0.018,rg:0.06, mg:0.085,beta:1.05,m12:0.12, m3:0.03, mc:23},
{t:'IMB.L',  n:'Imperial Brands',     s:'Consumer Staples',m:'UK',p:24.5,  pe:9,  pb:99,  roe:0.80,de:6.00,dy:0.082,rg:-0.02,mg:0.165,beta:0.45,m12:0.28, m3:0.07, mc:21},
{t:'BATS.L', n:'British American Tobacco',s:'Consumer Staples',m:'UK',p:24.0,pe:7,pb:2.2,roe:0.32,de:1.80,dy:0.090,rg:-0.03,mg:0.295,beta:0.50,m12:-0.08,m3:-0.02,mc:55},
{t:'PRU.L',  n:'Prudential plc',      s:'Financials',      m:'UK',p:8.50,  pe:14, pb:1.8, roe:0.14,de:1.00,dy:0.028,rg:0.06, mg:0.155,beta:0.85,m12:0.12, m3:0.03, mc:22},
{t:'SGRO.L', n:'Segro REIT',          s:'Real Estate',     m:'UK',p:8.20,  pe:25, pb:1.5, roe:0.06,de:0.60,dy:0.035,rg:0.08, mg:0.420,beta:0.80,m12:0.08, m3:0.02, mc:11},
{t:'LAND.L', n:'Land Securities',     s:'Real Estate',     m:'UK',p:6.80,  pe:12, pb:0.75,roe:0.07,de:0.55,dy:0.050,rg:0.03, mg:0.380,beta:0.90,m12:0.04, m3:0.01, mc:5},
{t:'CRH',    n:'CRH plc',             s:'Materials',       m:'IE',p:78.0,  pe:22, pb:3.5, roe:0.16,de:0.55,dy:0.017,rg:0.14, mg:0.115,beta:1.10,m12:0.32, m3:0.08, mc:55},
{t:'A5G.IR', n:'AIB Group plc',       s:'Financials',      m:'IE',p:5.20,  pe:7,  pb:0.85,roe:0.13,de:0.90,dy:0.055,rg:0.12, mg:0.310,beta:1.10,m12:0.42, m3:0.11, mc:12},
{t:'BIRG.IR', n:'Bank of Ireland',    s:'Financials',      m:'IE',p:9.80,  pe:8,  pb:0.90,roe:0.12,de:0.88,dy:0.048,rg:0.10, mg:0.280,beta:1.05,m12:0.38, m3:0.09, mc:8},
{t:'KRZ.IR', n:'Kerry Group plc',     s:'Consumer Staples',m:'IE',p:88.0,  pe:22, pb:3.2, roe:0.15,de:0.65,dy:0.012,rg:0.05, mg:0.120,beta:0.65,m12:0.04, m3:0.01, mc:15},
{t:'RYAAY',  n:'Ryanair Holdings',    s:'Industrials',     m:'IE',p:19.5,  pe:14, pb:4.5, roe:0.32,de:1.20,dy:0.000,rg:0.24, mg:0.180,beta:1.30,m12:0.28, m3:0.07, mc:22},
{t:'ICON.IR',n:'ICON plc',            s:'Healthcare',      m:'IE',p:262.0, pe:24, pb:3.5, roe:0.15,de:0.45,dy:0.000,rg:0.07, mg:0.130,beta:0.85,m12:0.08, m3:0.02, mc:19},
{t:'GFI.IR', n:'Glanbia plc',         s:'Consumer Staples',m:'IE',p:14.2,  pe:16, pb:2.5, roe:0.16,de:0.58,dy:0.018,rg:0.06, mg:0.073,beta:0.75,m12:0.12, m3:0.03, mc:5},
{t:'DRVT.IR',n:'DraftKings (Irish)',  s:'Consumer Disc.',  m:'IE',p:38.0,  pe:99, pb:5.5, roe:-0.12,de:0.20,dy:0.000,rg:0.35, mg:-0.055,beta:1.85,m12:0.45,m3:0.12, mc:19},
{t:'ORK.IR', n:'Ormonde Mining / Ornua',s:'Materials',     m:'IE',p:2.80,  pe:18, pb:1.8, roe:0.11,de:0.25,dy:0.010,rg:0.04, mg:0.085,beta:0.70,m12:0.06, m3:0.02, mc:1},
{t:'PTSB.IR',n:'Permanent TSB',       s:'Financials',      m:'IE',p:1.92,  pe:9,  pb:0.75,roe:0.09,de:0.92,dy:0.030,rg:0.08, mg:0.240,beta:1.15,m12:0.24, m3:0.06, mc:2},
{t:'INM.IR', n:'Independent News & Media',s:'Media',       m:'IE',p:0.12,  pe:12, pb:1.2, roe:0.10,de:0.30,dy:0.025,rg:-0.02,mg:0.065,beta:0.55,m12:-0.04,m3:-0.01,mc:1},
{t:'TOTAL.IR',n:'TotalEnergies (Dublin)',s:'Energy',       m:'IE',p:62.0,  pe:10, pb:1.4, roe:0.15,de:0.45,dy:0.048,rg:0.04, mg:0.115,beta:0.85,m12:0.10, m3:0.03, mc:145},
{t:'FBD.IR', n:'FBD Holdings',        s:'Financials',      m:'IE',p:14.5,  pe:8,  pb:1.1, roe:0.14,de:0.40,dy:0.045,rg:0.08, mg:0.145,beta:0.60,m12:0.22, m3:0.06, mc:1},
{t:'SMUR.IR',n:'Smurfit Kappa',       s:'Materials',       m:'IE',p:42.0,  pe:16, pb:2.8, roe:0.18,de:0.90,dy:0.030,rg:0.08, mg:0.105,beta:0.95,m12:0.18, m3:0.05, mc:23},
{t:'CPL.IR', n:'CPL Resources',       s:'Industrials',     m:'IE',p:11.5,  pe:12, pb:2.0, roe:0.17,de:0.15,dy:0.022,rg:0.06, mg:0.055,beta:0.70,m12:0.08, m3:0.02, mc:1},
{t:'TRIL.IR',n:'Trilogy Technologies',s:'Technology',      m:'IE',p:3.40,  pe:20, pb:2.5, roe:0.13,de:0.20,dy:0.000,rg:0.12, mg:0.115,beta:0.90,m12:0.14, m3:0.04, mc:1},
];

const PROFILES={
  value:    {l:'Value (Buffett)',     w:{pe:-0.28,pb:-0.20,roe:0.25,de:-0.15,dy:0.06,rg:0.04,mg:0.10,m12:0.04,m3:0.00}},
  growth:   {l:'Growth (Wood)',       w:{pe:-0.04,pb:-0.04,roe:0.15,de:-0.05,dy:-0.06,rg:0.42,mg:0.20,m12:0.14,m3:0.10}},
  dividend: {l:'Dividend Income',     w:{pe:-0.10,pb:-0.10,roe:0.20,de:-0.15,dy:0.38,rg:0.04,mg:0.10,m12:0.04,m3:0.00}},
  momentum: {l:'Momentum Trader',     w:{pe:0.00,pb:0.00,roe:0.05,de:-0.05,dy:0.00,rg:0.10,mg:0.04,m12:0.48,m3:0.32}},
  quality:  {l:'QARP',                w:{pe:-0.15,pb:-0.10,roe:0.30,de:-0.15,dy:0.05,rg:0.15,mg:0.22,m12:0.05,m3:0.00}},
  macro:    {l:'Global Macro',        w:{pe:-0.12,pb:-0.08,roe:0.18,de:-0.20,dy:0.10,rg:0.15,mg:0.12,m12:0.12,m3:0.08}},
  activist: {l:'Activist/Deep Value', w:{pe:-0.35,pb:-0.30,roe:0.15,de:-0.10,dy:0.08,rg:0.02,mg:0.08,m12:-0.04,m3:-0.02}},
};
const RISK={low:{bp:0.35,mb:0.8},medium:{bp:0.10,mb:1.5},high:{bp:0.00,mb:99}};

let activeMkt='US';
let selectedTickers=new Set(['AAPL','MSFT','NVDA','JPM','BRK-B','META','JNJ','PG','SHEL.L','AZN.L','CRH','A5G.IR','RYAAY']);
const customStocks=new Map(); // ticker → {t,n,s,m} for tickers added via search
let charts={};
let runResults=null;

function getStockMeta(tk){
  return ALL_STOCKS.find(s=>s.t===tk)||customStocks.get(tk)||{t:tk,n:tk,s:'Unknown',m:'US'};
}

function debounce(fn,ms){let timer;return(...a)=>{clearTimeout(timer);timer=setTimeout(()=>fn(...a),ms);};}

function addCustomStock(ticker,name){
  if(selectedTickers.size>=40)return;
  customStocks.set(ticker,{t:ticker,n:name,s:'Unknown',m:'US'});
  selectedTickers.add(ticker);
  // Refresh add button states in search results
  document.querySelectorAll('#sp-searchResultsList .sp-sr-add').forEach(btn=>{
    const row=btn.closest('.sp-search-result-row');
    if(row&&row.querySelector('.sp-sr-ticker')?.textContent===ticker){
      btn.textContent='Added';btn.classList.add('selected');btn.disabled=true;
    }
  });
  renderSelectedList();
  initStockList();
}

function initStockList(){
  const list=document.getElementById('stockList');
  list.innerHTML='';
  const arr=[...selectedTickers];
  arr.forEach(tk=>{
    const s=getStockMeta(tk);
    const div=document.createElement('div');
    div.className='stock-item active';
    div.dataset.ticker=tk;
    const mc=s.m==='US'?'mkt-us':s.m==='UK'?'mkt-uk':'mkt-ie';
    div.innerHTML=`<div class="si-ticker">${esc(tk)}</div><div class="si-info"><div class="si-name">${esc(s.n)}</div><div class="si-sector">${esc(s.s)} <span class="sr-mkt ${mc}" style="padding:1px 5px">${s.m}</span></div></div><button class="si-remove" data-tk="${esc(tk)}">×</button>`;
    div.querySelector('.si-remove').onclick=e=>{e.stopPropagation();selectedTickers.delete(tk);initStockList();};
    list.appendChild(div);
  });
  document.getElementById('selCount').textContent=`(${arr.length})`;
}

function addStock(tk){
  selectedTickers.add(tk);
  initStockList();
  document.getElementById('searchInput').value='';
  document.getElementById('searchResults').style.display='none';
}

function doSearch(q){
  const sr=document.getElementById('searchResults');
  if(!q){sr.style.display='none';return;}
  const pool=activeMkt==='ALL'?ALL_STOCKS:ALL_STOCKS.filter(s=>s.m===activeMkt);
  const lq=q.toLowerCase();
  const hits=pool.filter(s=>s.t.toLowerCase().includes(lq)||s.n.toLowerCase().includes(lq)).slice(0,9);
  if(hits.length){
    sr.innerHTML=hits.map(s=>{
      const mc=s.m==='US'?'mkt-us':s.m==='UK'?'mkt-uk':'mkt-ie';
      const added=selectedTickers.has(s.t);
      return`<div class="sr-item" data-tk="${esc(s.t)}" style="${added?'opacity:0.4;pointer-events:none':''}">
        <span class="sr-ticker">${highlightMatch(s.t,q)}</span>
        <span style="flex:1;font-size:11px;color:var(--text2)">${highlightMatch(s.n,q)}</span>
        <span class="sr-mkt ${mc}">${s.m}</span>
        ${added?'<span style="font-size:9px;color:var(--text3)">added</span>':''}
      </div>`;
    }).join('');
    sr.style.display='block';
    sr.querySelectorAll('.sr-item:not([style*="pointer-events:none"])').forEach(el=>{el.onclick=()=>addStock(el.dataset.tk);});
  } else {
    // No local match — fall back to Finnhub for tickers outside the curated list
    sr.style.display='none';
    _debouncedSidebarSearch(q);
  }
}

// Sidebar Finnhub fallback — only fires when ALL_STOCKS has no match.
// Uses the shared SearchCache so results are never fetched twice.
const _debouncedSidebarSearch=debounce(async function(query){
  const sr=document.getElementById('searchResults');
  const cacheKey='sb:'+query.toLowerCase();
  let hits=SearchCache.get(cacheKey);
  if(!hits){
    try{
      const r=await fetch(`/api/search?q=${encodeURIComponent(query)}&types=stock`);
      if(!r.ok)return;
      const d=await r.json();
      hits=(d.results||[]).slice(0,9);
      SearchCache.set(cacheKey,hits);
    }catch{return;}
  }
  if(!hits.length)return;
  sr.innerHTML=hits.map(h=>{
    const added=selectedTickers.has(h.ticker);
    return`<div class="sr-item" data-tk="${esc(h.ticker)}" style="${added?'opacity:0.4;pointer-events:none':''}">
      <span class="sr-ticker">${highlightMatch(h.ticker,query)}</span>
      <span style="flex:1;font-size:11px;color:var(--text2)">${highlightMatch(h.name,query)}</span>
      <span class="sr-mkt mkt-us">US</span>
      ${added?'<span style="font-size:9px;color:var(--text3)">added</span>':''}
    </div>`;
  }).join('');
  sr.style.display='block';
  sr.querySelectorAll('.sr-item:not([style*="pointer-events:none"])').forEach(el=>{el.onclick=()=>addStock(el.dataset.tk);});
},400);

document.getElementById('searchInput').addEventListener('input',e=>doSearch(e.target.value));
document.getElementById('searchInput').addEventListener('focus',e=>{if(e.target.value)doSearch(e.target.value);});
document.addEventListener('click',e=>{if(!e.target.closest('.search-wrap'))document.getElementById('searchResults').style.display='none';});

document.querySelectorAll('.mkt-tab').forEach(tab=>{
  tab.onclick=()=>{
    activeMkt=tab.dataset.mkt;
    document.querySelectorAll('.mkt-tab').forEach(t=>t.classList.remove('active'));
    tab.classList.add('active');
    doSearch(document.getElementById('searchInput').value);
  };
});

function sb(id,vid,fmt){const s=document.getElementById(id),v=document.getElementById(vid);s.oninput=()=>v.textContent=fmt(+s.value);v.textContent=fmt(+s.value);}
sb('lbSlider','lbVal',v=>`${v} yr`);
sb('lamSlider','lamVal',v=>(v/100).toFixed(2));
sb('depthSlider','depthVal',v=>v);
sb('foldSlider','foldVal',v=>v);

function norm(vals){
  const mn=Math.min(...vals),mx=Math.max(...vals);
  if(mx===mn)return vals.map(()=>0.5);
  return vals.map(v=>(v-mn)/(mx-mn));
}

function factorProfile(stocks){
  return stocks.map(s=>({
    value:      s.pe>0&&(1/s.pe)>0.06?0.03:(s.pe>0&&(1/s.pe)<0.02)?-0.02:0,
    quality:    s.mg>0.20?0.025:s.mg<0.08?-0.015:0.005,
    investment: s.rg<0.05?0.015:s.rg>0.20?-0.020:0,
    size:       s.mc<100?0.040:s.mc<400?0.010:-0.010,
    momentum:   s.m12>0.20?0.030:s.m12<-0.05?-0.025:0.005,
  }));
}

function computeScores(stocks,profile,risk,lambda,useFF5){
  const w=profile.w,rp=RISK[risk];
  const feats=['pe','pb','roe','de','dy','rg','mg','m12','m3'];
  const nd={};feats.forEach(f=>{nd[f]=norm(stocks.map(s=>s[f]));});
  const fp=factorProfile(stocks);
  return stocks.map((s,i)=>{
    let raw=0;
    feats.forEach(f=>{raw+=(w[f]||0)*nd[f][i];});
    const betaPen=rp.bp*Math.max(0,s.beta-rp.mb);
    const l2=lambda*0.05;
    let fund=Math.max(0,Math.min(1,raw-betaPen-l2+0.42));
    const q=useFF5?(fp[i].value+fp[i].quality+fp[i].investment+fp[i].momentum)*0.35:0;
    const composite=fund+q;
    const noise=(Math.random()-0.5)*0.028;
    const tilt=fp[i].value+fp[i].quality+fp[i].investment+fp[i].size+fp[i].momentum;
    return{
      ...s,
      score:     Math.max(0.04,Math.min(0.97,composite+noise)),
      fundScore: +fund.toFixed(3),
      quantScore:+(tilt*100/5+50).toFixed(1),
      ff:        fp[i],
    };
  });
}

function getSignal(score){
  if(score>0.70)return'STRONG BUY';
  if(score>0.54)return'BUY';
  if(score>0.38)return'HOLD';
  return'SELL';
}
function isBuy(signal){return signal==='STRONG BUY'||signal==='BUY';}
function pillCls(sig){
  if(sig==='STRONG BUY')return'p-sb';
  if(sig==='BUY')return'p-b';
  if(sig==='HOLD')return'p-h';
  return'p-s';
}

function buildBuyReasons(s,profile){
  const reasons=[];
  const w=profile.w;
  const feats=[
    {k:'pe',  lbl:'attractive valuation',   good: s.pe<18,       val:`P/E ${s.pe}x`},
    {k:'pb',  lbl:'low price-to-book',       good: s.pb<2.5,      val:`P/B ${s.pb}x`},
    {k:'roe', lbl:'strong return on equity', good: s.roe>0.2,     val:`ROE ${(s.roe*100).toFixed(0)}%`},
    {k:'mg',  lbl:'high net margin',         good: s.mg>0.15,     val:`margin ${(s.mg*100).toFixed(1)}%`},
    {k:'rg',  lbl:'revenue growth',          good: s.rg>0.1,      val:`+${(s.rg*100).toFixed(0)}% revenue`},
    {k:'de',  lbl:'conservative leverage',   good: s.de<0.6,      val:`D/E ${s.de.toFixed(2)}x`},
    {k:'dy',  lbl:'dividend yield',          good: s.dy>0.02,     val:`${(s.dy*100).toFixed(1)}% yield`},
    {k:'m12', lbl:'positive momentum',       good: s.m12>0.1,     val:`+${(s.m12*100).toFixed(0)}% 12M`},
    {k:'beta',lbl:'low market risk',         good: s.beta<0.8,    val:`β ${s.beta.toFixed(2)}`},
  ];
  feats.filter(f=>f.good&&Math.abs(w[f.k]||0)>0.03).forEach(f=>{reasons.push(`${f.val} — ${f.lbl}`);});
  return reasons.slice(0,4);
}

function buildSellReasons(s,profile){
  const reasons=[];
  if(s.pe>50)    reasons.push(`Stretched valuation at P/E ${s.pe}x — requires perfect execution`);
  if(s.m12<-0.08)reasons.push(`Negative 12M momentum (${(s.m12*100).toFixed(0)}%) — price trend is deteriorating`);
  if(s.de>2.0)   reasons.push(`High leverage at ${s.de.toFixed(2)}x D/E — elevated refinancing risk`);
  if(s.rg<0.0)   reasons.push(`Declining revenues — structural headwinds or competitive pressure`);
  if(s.mg<0.05)  reasons.push(`Very thin margins — limited operational resilience`);
  return reasons.slice(0,4);
}

async function runAnalysis(){
  const tickers=[...selectedTickers];
  if(tickers.length<3){alert('Please select at least 3 stocks.');return;}
  const profileKey=document.getElementById('sp-profileSel').value;
  const riskKey=spRiskKey;
  const lambda=+document.getElementById('lamSlider').value/100;
  const folds=+document.getElementById('foldSlider').value;
  const useFF5=document.getElementById('quantToggle').checked;
  const useGeo=document.getElementById('geoToggle').checked;
  const profile=PROFILES[profileKey];

  const overlay=document.getElementById('loaderOverlay');
  const stepsEl=document.getElementById('loaderSteps');
  overlay.style.display='flex';

  const backendLive=await checkBackend();

  let steps;
  if(backendLive){
    steps=['Connecting to NUMKT ML backend','Fetching live market data (Finnhub / Yahoo Finance)',`Running ${folds}-fold TimeSeriesSplit CV`,'Training Random Forest (300 trees)','Training Gradient Boosting ensemble','Blending predictions + factor signals','Generating BUY / HOLD / SELL signals'];
  } else {
    steps=['Backend offline — using local factor model',`Running ${folds}-fold cross-validation (simulated)`,'Training Random Forest (500 trees)','Training Gradient Boosting ensemble','Blending model predictions'];
    if(useFF5)steps.push('Computing factor profile signals');
    steps.push('Generating BUY / HOLD / SELL signals');
  }

  stepsEl.innerHTML=steps.map((s,i)=>`<div class="ls" id="ls${i}"><div class="ls-dot"></div>${s}</div>`).join('');

  let backendResults=null,cvAccuracy=null,cvIC=null,featureImportance=null,analyseStatus=null;

  if(backendLive){
    document.getElementById('ls0').className='ls active';
    await new Promise(r=>setTimeout(r,400));
    document.getElementById('ls0').className='ls done';
    document.getElementById('ls1').className='ls active';
    const resp=await callBackend('/analyse',{tickers,profile:profileKey,risk:riskKey,lookback_years:+document.getElementById('sp-lbSlider').value,cv_folds:folds,lambda_reg:lambda});
    analyseStatus=resp.status;
    backendResults=resp.ok?resp.data:null;
    for(let i=2;i<steps.length;i++){
      document.getElementById(`ls${i-1}`).className='ls done';
      document.getElementById(`ls${i}`).className='ls active';
      await new Promise(r=>setTimeout(r,280+Math.random()*200));
    }
    document.getElementById(`ls${steps.length-1}`).className='ls done';
    if(backendResults){cvAccuracy=backendResults.cv_accuracy;cvIC=backendResults.cv_ic??null;featureImportance=backendResults.feature_importance;}
  }
  await new Promise(r=>setTimeout(r,180));

  // No live data → show an honest state. Never render stale/hardcoded prices.
  if(!(backendResults&&backendResults.results&&backendResults.results.length)){
    overlay.style.display='none';
    updateBackendBadge(false);
    if(analyseStatus===401||!Auth.isLoggedIn()){
      _showToast('Log in to run a live analysis with real market data.', 'info');
      Auth.openModal();
    } else {
      _showToast('Couldn’t fetch live market data right now — please try again in a moment.', 'err');
    }
    return;
  }

  const scored=backendResults.results.map(br=>{
    const local=ALL_STOCKS.find(s=>s.t===br.ticker)||customStocks.get(br.ticker)||{};
    // Days until next earnings (null if not available or already passed)
    let daysToEarnings=null;
    if(br.next_earnings_date){
      const diff=Math.round((new Date(br.next_earnings_date)-Date.now())/(1000*60*60*24));
      if(diff>=0&&diff<=90)daysToEarnings=diff;
    }
    return{...local,t:br.ticker,n:br.name||local.n||br.ticker,s:br.sector||local.s||'Unknown',m:br.market||local.m||'US',p:br.last_price??null,score:br.composite_score/100,fundScore:br.fundamental_score/100,instPct:br.inst_ownership_pct,insiderPct:br.insider_ownership_pct,ff:mapFactorProfile(br.factor_profile),signal:br.signal,buyReasons:br.buy_reasons||[],pe:br.fundamentals?.pe_ratio??local.pe,roe:br.fundamentals?.roe!=null?br.fundamentals.roe/100:local.roe,mg:br.fundamentals?.net_margin!=null?br.fundamentals.net_margin/100:local.mg,de:br.fundamentals?.debt_equity??local.de,dy:br.fundamentals?.dividend_yield!=null?br.fundamentals.dividend_yield/100:local.dy,rg:br.fundamentals?.revenue_growth!=null?br.fundamentals.revenue_growth/100:local.rg,m12:br.fundamentals?.mom_12m!=null?br.fundamentals.mom_12m/100:local.m12,beta:br.fundamentals?.beta??local.beta,betaNote:br.beta_note||null,mc:br.market_cap_bn??local.mc,isLive:true,analystBuy:br.analyst_buy??null,analystHold:br.analyst_hold??null,analystSell:br.analyst_sell??null,daysToEarnings};
  });

  updateBackendBadge(true);
  runResults={scored,profileKey,riskKey,profile,useFF5,folds,lambda,cvAccuracy,cvIC,featureImportance,backendLive:true};
  overlay.style.display='none';
  renderResults(runResults);
}

function mapFactorProfile(fp){
  if(!fp)return{value:0,quality:0,investment:0,size:0,momentum:0};
  return{value:(fp.value||0)/100,quality:(fp.quality||0)/100,investment:(fp.investment||0)/100,size:(fp.size||0)/100,momentum:(fp.momentum||0)/100};
}

function updateBackendBadge(live){
  const hdr=document.querySelector('.header-right');
  if(!hdr)return;
  let badge=document.getElementById('backendBadge');
  if(!badge){badge=document.createElement('div');badge.id='backendBadge';badge.className='hbadge';hdr.insertBefore(badge,hdr.firstChild);}
  if(live){badge.textContent='LIVE DATA';badge.style.cssText='background:var(--green-dim);color:var(--green);border:1px solid var(--green-border)';}
  else{badge.textContent='OFFLINE';badge.style.cssText='background:var(--surface);color:var(--text3);border:1px solid var(--border2)';}
}

function renderResults(R){
  const{scored,profileKey,riskKey,profile,useFF5,folds,lambda,cvIC}=R;
  showPage('results');

  // Always refresh the save-name prefill so each new analysis gets its own default
  const nameInput = document.getElementById('saveNameInput');
  if (nameInput) {
    const top3 = scored.slice(0, 3).map(s => s.t).join(', ');
    nameInput.value = `${profile.l} — ${top3}`;
  }

  const buys=scored.filter(s=>isBuy(s.signal));
  const holds=scored.filter(s=>s.signal==='HOLD');
  const sells=scored.filter(s=>s.signal==='SELL');
  const avgS=(scored.reduce((a,b)=>a+b.score,0)/scored.length*100).toFixed(0);
  const top=scored[0];
  const cvAcc=R.cvAccuracy!=null?(R.cvAccuracy*100).toFixed(1):null;
  const cvICVal=cvIC!=null?cvIC.toFixed(4):null;
  const mkts=[...new Set(scored.map(s=>s.m))].join(' / ');

  document.getElementById('summaryGrid').innerHTML=`
    <div class="metric-card hl animate-in"><div class="mc-label">Top Pick</div><div class="mc-val">${top.t}</div><div class="mc-sub">${top.signal} · ${(top.score*100).toFixed(0)}%</div></div>
    <div class="metric-card hl-green animate-in delay-1"><div class="mc-label">BUY Signals</div><div class="mc-val">${buys.length}</div><div class="mc-sub">${buys.slice(0,3).map(s=>s.t).join(', ')}</div></div>
    <div class="metric-card animate-in delay-2"><div class="mc-label">HOLD</div><div class="mc-val">${holds.length}</div><div class="mc-sub">of ${scored.length} stocks · ${mkts}</div></div>
    <div class="metric-card animate-in delay-3"><div class="mc-label">Model IC</div><div class="mc-val">${cvICVal??'—'}</div><div class="mc-sub">${cvAcc!=null?`acc ${cvAcc}% · `:''}${folds}-fold CV</div></div>
    <div class="metric-card animate-in delay-4"><div class="mc-label">Avg Confidence</div><div class="mc-val">${avgS}%</div><div class="mc-sub">${profile.l}</div></div>`;

  const bb=document.getElementById('buyBannerSlot');
  if(buys.length){
    const cards=buys.slice(0,6).map(s=>`
      <div class="bb-card">
        <div class="bb-ticker">${esc(s.t)}${s.signal==='STRONG BUY'?'<span style="font-size:8px;font-family:\'JetBrains Mono\',monospace;background:var(--green-dim);border:1px solid var(--green-border);color:var(--green);padding:2px 5px;border-radius:9999px;margin-left:6px;vertical-align:middle">STRONG</span>':''}</div>
        <div class="bb-name">${esc(s.n)}</div>
        <div class="bb-score-row">
          <span class="bb-score">${(s.score*100).toFixed(0)}%</span>
          <div class="bb-bar"><div class="bb-fill" style="width:${(s.score*100).toFixed(0)}%"></div></div>
        </div>
        <div class="bb-reason">${s.buyReasons[0]||''}</div>
      </div>`).join('');
    bb.innerHTML=`<div class="buy-banner animate-in delay-1">
      <div class="bb-title"><div class="bb-pulse"></div>MODEL BUY RECOMMENDATIONS — ${buys.length} signal${buys.length>1?'s':''} identified</div>
      <div class="bb-cards">${cards}</div>
    </div>`;
  } else bb.innerHTML='';

  document.getElementById('hfPanelSlot').innerHTML='';

  const qp=document.getElementById('quantPanelSlot');
  if(useFF5){
    const avg=arr=>+(arr.reduce((a,b)=>a+b,0)/arr.length*100).toFixed(2);
    const val=avg(scored.map(s=>s.ff.value));
    const ql =avg(scored.map(s=>s.ff.quality));
    const inv=avg(scored.map(s=>s.ff.investment));
    const sz =avg(scored.map(s=>s.ff.size));
    const mom=avg(scored.map(s=>s.ff.momentum));
    qp.innerHTML=`<div class="qt-panel animate-in delay-1"><div class="qt-title">Factor Profile (Universe Average)</div>
      <div class="qt-metrics">
        <div class="qt-metric"><div class="qt-lbl">Value</div><div class="qt-val" style="color:${val>0?'var(--green)':'var(--red)'}">${val>0?'+':''}${val}%</div></div>
        <div class="qt-metric"><div class="qt-lbl">Quality</div><div class="qt-val" style="color:${ql>0?'var(--green)':'var(--red)'}">${ql>0?'+':''}${ql}%</div></div>
        <div class="qt-metric"><div class="qt-lbl">Investment</div><div class="qt-val" style="color:${inv>0?'var(--green)':'var(--red)'}">${inv>0?'+':''}${inv}%</div></div>
        <div class="qt-metric"><div class="qt-lbl">Size</div><div class="qt-val" style="color:${sz>0?'var(--green)':'var(--red)'}">${sz>0?'+':''}${sz}%</div></div>
        <div class="qt-metric"><div class="qt-lbl">Momentum</div><div class="qt-val" style="color:${mom>0?'var(--green)':'var(--red)'}">${mom>0?'+':''}${mom}%</div></div>
      </div></div>`;
  } else qp.innerHTML='';

  const top3=buys.slice(0,3).map(s=>s.t).join(', ')||top.t;
  const insightMap={
    value:`The value model identifies ${top3} as the strongest candidates — each trading at a meaningful discount to intrinsic value with durable returns on capital well above the cost of equity. The model weights P/E, price-to-book, and ROE most heavily, with L2 regularisation preventing single-factor over-reliance.`,
    growth:`Growth screening surfaces ${top3} as high-conviction compounders — demonstrating accelerating revenue with expanding margins that justify premium multiples. The model penalises overleveraged names and rewards consistent top-line momentum.`,
    dividend:`Income screening prioritises sustainable yield with quality protection. ${top3} lead — combining attractive dividend yields with strong balance sheets and cash flow cover. The model filters out yield traps by requiring minimum ROE and margin thresholds.`,
    momentum:`Momentum filters identify ${top3} as exhibiting the strongest risk-adjusted trend persistence across 3-month and 12-month windows. Cross-validation confirms these signals are not artefacts of overfitting.`,
    quality:`QARP discipline surfaces ${top3} — exceptional capital efficiency at prices a rational long-term investor can defend. The model blends ROE, net margin, and leverage discipline with a valuation anchor.`,
    macro:`The global macro screen identifies ${top3} as best positioned given the current rate, FX, and growth regime. Rate-sensitive sectors are penalised under the model's risk-adjusted scoring.`,
    activist:`Deep value screening surfaces ${top3} as candidates trading at steep discounts with identifiable catalysts for re-rating — low P/B, depressed multiples, and potential for operational improvement.`,
  };
  document.getElementById('insightBox').innerHTML=`
    <div class="animate-in">
      <div class="ib-title">Model Summary — ${profile.l} · ${riskKey} risk · ${scored.length} stocks analysed</div>
      <div class="ib-text">${insightMap[profileKey]||insightMap.value}</div>
    </div>`;

  document.getElementById('recTable').innerHTML=`
    <div class="rec-head"><span>Ticker</span><span>Company</span><span>Score</span><span>P/E</span><span>ROE</span><span>Beta</span><span>Analysts</span><span>Signal</span></div>
    ${scored.map(s=>{
      const mc=s.m==='US'?'mkt-us':s.m==='UK'?'mkt-uk':'mkt-ie';
      const buyRow=isBuy(s.signal)?'is-buy':'';
      const earningsBadge=s.daysToEarnings!=null
        ?`<span class="earnings-badge" title="Next earnings report">⚠ ${s.daysToEarnings===0?'Today':s.daysToEarnings+'d'}</span>`:'';
      const analystStr=(s.analystBuy!=null||s.analystHold!=null||s.analystSell!=null)
        ?`<div class="analyst-row"><span class="ab">${s.analystBuy??0}B</span> · <span class="ah">${s.analystHold??0}H</span> · <span class="as">${s.analystSell??0}S</span></div>`
        :`<div class="analyst-row" style="color:var(--text3)">—</div>`;
      return`<div class="rec-row ${buyRow}" data-ticker="${s.t}">
        <span class="rt-ticker">${s.t} <span class="sr-mkt ${mc}" style="font-size:8px;padding:1px 4px">${s.m}</span>${s.isLive?'<span style="font-size:8px;color:var(--teal);margin-left:3px" title="Live">●</span>':''}</span>
        <span class="rt-name">${esc(s.n)}${isBuy(s.signal)?'<span class="buy-tag">BUY</span>':''}${earningsBadge}</span>
        <span><div class="rt-num">${(s.score*100).toFixed(0)}%</div><div class="conf-bar"><div class="conf-fill" style="width:${(s.score*100).toFixed(0)}%;background:${s.score>0.65?'var(--green)':s.score>0.45?'var(--amber)':'var(--red)'}"></div></div></span>
        <span class="rt-num">${s.pe>90?'N/A':s.pe+'x'}</span>
        <span class="rt-num">${(s.roe*100).toFixed(0)}%</span>
        <span class="rt-num">${s.beta.toFixed(1)}</span>
        <span>${analystStr}</span>
        <span><span class="sig-pill ${pillCls(s.signal)}">${s.signal}</span></span>
      </div>`;
    }).join('')}`;
  document.querySelectorAll('.rec-row').forEach(r=>{r.onclick=()=>openDetail(r.dataset.ticker);});

  renderCharts(scored,profileKey,R.featureImportance);
  document.getElementById('disclaimer').textContent='NUMKT is for educational and informational purposes only. The ML model uses simulated historical factors and is not connected to live market data. Signals do not constitute financial advice. Past simulated performance does not predict future results. Always consult a qualified financial adviser before making any investment decision.';
}

function renderCharts(scored,profileKey,backendFI){
  Object.values(charts).forEach(c=>c.destroy());
  charts={};
  Chart.defaults.color='#52525B';
  Chart.defaults.font.family="'JetBrains Mono', monospace";

  const labels=scored.map(s=>s.t);
  const scoreData=scored.map(s=>+(s.score*100).toFixed(0));
  const colors=scored.map(s=>s.score>0.65?'rgba(52,210,123,0.7)':s.score>0.45?'rgba(240,160,48,0.7)':'rgba(240,84,84,0.7)');
  charts.dist=new Chart(document.getElementById('distChart'),{
    type:'bar',
    data:{labels,datasets:[{data:scoreData,backgroundColor:colors,borderRadius:4,borderSkipped:false}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
      scales:{y:{min:0,max:100,ticks:{color:'#4e4a5e',font:{size:9}},grid:{color:'rgba(255,255,255,0.04)'}},
              x:{ticks:{color:'#4e4a5e',font:{size:9},maxRotation:55,autoSkip:false},grid:{display:false}}}}
  });

  const months=['Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar'];
  const top5=scored.slice(0,5);
  const dsCol=['#84CC16','#6080f5','#34d27b','#f05454','#a585f0'];
  charts.mom=new Chart(document.getElementById('momChart'),{
    type:'line',
    data:{labels:months,datasets:top5.map((s,i)=>{let c=100;return{label:s.t,data:months.map(()=>{c*=(1+(s.m12/12)*(0.6+Math.random()*0.8));return+c.toFixed(1);}),borderColor:dsCol[i],backgroundColor:'transparent',tension:0.4,pointRadius:0,borderWidth:2};})},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:true,labels:{color:'#9d99a8',font:{size:9},boxWidth:8,padding:7}}},
      scales:{y:{ticks:{color:'#4e4a5e',font:{size:9}},grid:{color:'rgba(255,255,255,0.04)'}},x:{ticks:{color:'#4e4a5e',font:{size:9}},grid:{display:false}}}}
  });

  const grps={};
  scored.forEach(s=>{const k=`${s.m}:${s.s.slice(0,12)}`;grps[k]=(grps[k]||0)+1;});
  const gKeys=Object.keys(grps),gData=Object.values(grps);
  const gCol=['#84CC16','#6080f5','#34d27b','#f05454','#a585f0','#30d8c8','#f0a030','#60a0f0','#d08060','#60a860','#a05040','#40a090'];
  charts.sec=new Chart(document.getElementById('secChart'),{
    type:'doughnut',
    data:{labels:gKeys,datasets:[{data:gData,backgroundColor:gCol.slice(0,gKeys.length),borderWidth:0}]},
    options:{responsive:true,maintainAspectRatio:false,cutout:'60%',plugins:{legend:{display:true,position:'right',labels:{color:'#9d99a8',font:{size:9},boxWidth:8,padding:5}}}}
  });

  if(backendFI&&backendFI.length){
    const maxFI=Math.max(...backendFI.map(f=>f.importance));
    document.getElementById('fiGrid').innerHTML=backendFI.slice(0,10).map(f=>{
      const pct=maxFI>0?(f.importance/maxFI*100).toFixed(0):0;
      const rfPct=maxFI>0?(f.rf/maxFI*100).toFixed(0):0;
      const gbPct=maxFI>0?(f.gb/maxFI*100).toFixed(0):0;
      return`<div class="fi-row"><div class="fi-label"><span>${f.feature}</span><span style="color:var(--teal);font-size:9px">RF:${rfPct}% GB:${gbPct}%</span></div><div class="fi-bar"><div class="fi-fill" style="width:${pct}%;background:var(--teal)"></div></div></div>`;
    }).join('');
  } else {
    const w=PROFILES[profileKey].w;
    const feats=[{k:'rg',l:'Rev Growth'},{k:'mg',l:'Net Margin'},{k:'roe',l:'ROE'},{k:'pe',l:'P/E'},{k:'pb',l:'P/B'},{k:'m12',l:'12M Mom.'},{k:'m3',l:'3M Mom.'},{k:'de',l:'Debt/Eq.'},{k:'dy',l:'Div. Yield'},{k:'beta',l:'Beta'}];
    const maxW=Math.max(...feats.map(f=>Math.abs(w[f.k]||0)));
    document.getElementById('fiGrid').innerHTML=feats.map(f=>{
      const imp=Math.abs(w[f.k]||0);
      const pct=maxW>0?(imp/maxW*100).toFixed(0):0;
      const pos=(w[f.k]||0)>=0;
      return`<div class="fi-row"><div class="fi-label"><span>${f.l}</span><span style="color:${pos?'var(--green)':'var(--red)'};font-size:9px">${pos?'▲':'▼'} ${pct}%</span></div><div class="fi-bar"><div class="fi-fill" style="width:${pct}%;background:${pos?'var(--green)':'var(--red)'}"></div></div></div>`;
    }).join('');
  }
}

function openDetail(ticker){
  const r=runResults&&runResults.scored.find(x=>x.t===ticker);
  const staticS=ALL_STOCKS.find(x=>x.t===ticker)||{};
  const s={...staticS,...(r||{})};
  if(!s)return;

  const score=r?(r.score*100).toFixed(0):'—';
  const signal=r?r.signal:'—';
  const mc=s.m==='US'?'mkt-us':s.m==='UK'?'mkt-uk':'mkt-ie';
  const buy=isBuy(signal);
  const hasPrice=r&&r.p!=null;
  const priceStr=hasPrice
    ? (s.m==='UK'?'£':'')+(s.m==='US'?'$':'')+(s.m==='IE'?'€':'')+r.p.toLocaleString()
    : '—';

  const vClass=buy?'v-buy':signal==='HOLD'?'v-hold':'v-sell';
  const vLabel=buy?`BUY ${s.t}`:signal==='HOLD'?`HOLD ${s.t}`:`AVOID ${s.t}`;
  const vDesc=buy
    ?`The NUMKT model recommends buying ${s.t}. The composite ML score of ${score}% places this stock in the ${signal} tier based on fundamental analysis and quant factor decomposition.`
    :signal==='HOLD'
    ?`The model suggests holding ${s.t}. The stock shows mixed signals — some factors support it, others don't. Not a clear buy at current levels.`
    :`The model signals caution on ${s.t}. The composite score of ${score}% reflects weak fundamentals or negative momentum.`;
  const reasons=r?r.buyReasons:[];
  const rats=buildDetailRationale(s);

  document.getElementById('dpContent').innerHTML=`
    <div style="margin-bottom:1.1rem">
      <div class="dp-ticker">${s.t} <span class="sr-mkt ${mc}" style="font-size:11px;vertical-align:middle">${s.m}</span></div>
      <div class="dp-name">${esc(s.n)} · ${esc(s.s)}</div>
      <div class="dp-price">${priceStr}</div>
      <div style="margin-top:9px;display:flex;gap:7px;align-items:center;flex-wrap:wrap">
        <span class="sig-pill ${pillCls(signal)}">${signal}</span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text3)">Model score: ${score}%</span>
      </div>
    </div>

    <div class="verdict-box ${vClass}">
      <div class="v-headline">${vLabel}</div>
      <div style="font-size:11px;color:var(--text2);line-height:1.6;margin-bottom:4px">${vDesc}</div>
      ${reasons.length?`<ul class="v-reasons">${reasons.map(rs=>`<li class="v-reason"><span class="v-icon">${buy?'✓':'⚠'}</span><span>${rs}</span></li>`).join('')}</ul>`:''}
    </div>

    <div class="score-row">
      <div class="sc-box"><div class="sc-lbl">FUNDAMENTAL</div><div class="sc-v">${r?(r.fundScore*100).toFixed(0):'-'}</div></div>
      <div class="sc-box"><div class="sc-lbl">FACTOR</div><div class="sc-v" style="color:${r&&(r.ff.value+r.ff.quality+r.ff.investment+r.ff.momentum)>0?'var(--green)':'var(--red)'}">${r?r.quantScore:'—'}</div></div>
      <div class="sc-box"><div class="sc-lbl">COMPOSITE</div><div class="sc-v">${score}%</div></div>
    </div>

    <div class="dp-section">
      <div class="dp-stitle">Fundamental Factor Analysis</div>
      <ul class="rat-list">${rats.map(rt=>`<li class="rat-item ${rt.t}"><span class="rat-icon">${rt.i}</span><span>${rt.txt}</span></li>`).join('')}</ul>
    </div>

    ${r&&r.ff?`<div class="dp-section">
      <div class="dp-stitle">Factor Profile</div>
      <div class="ff5-box">
        <div class="ff5-title">Factor Signals</div>
        <div class="ff5-grid">
          <div class="ff5-row"><span>Value</span><span style="color:${r.ff.value>0?'var(--green)':'var(--red)'}">${r.ff.value>0?'+':''}${(r.ff.value*100).toFixed(2)}%</span></div>
          <div class="ff5-row"><span>Quality</span><span style="color:${r.ff.quality>0?'var(--green)':'var(--red)'}">${r.ff.quality>0?'+':''}${(r.ff.quality*100).toFixed(2)}%</span></div>
          <div class="ff5-row"><span>Investment</span><span style="color:${r.ff.investment>0?'var(--green)':'var(--red)'}">${r.ff.investment>0?'+':''}${(r.ff.investment*100).toFixed(2)}%</span></div>
          <div class="ff5-row"><span>Size</span><span style="color:${r.ff.size>0?'var(--green)':'var(--red)'}">${r.ff.size>0?'+':''}${(r.ff.size*100).toFixed(2)}%</span></div>
          <div class="ff5-row"><span>Momentum</span><span style="color:${r.ff.momentum>0?'var(--green)':'var(--red)'}">${r.ff.momentum>0?'+':''}${(r.ff.momentum*100).toFixed(2)}%</span></div>
        </div>
      </div>
    </div>`:''}

    <div class="dp-section">
      <div class="dp-stitle">Key Fundamentals</div>
      <table class="dp-table">
        <tr><td>Price</td><td>${priceStr}${r&&r.isLive?'<span style="font-size:9px;color:var(--teal);margin-left:5px">● live</span>':''}</td></tr>
        <tr><td>Market Cap</td><td>${s.mc?'$'+s.mc+'bn':'—'}</td></tr>
        <tr><td>P/E Ratio</td><td>${s.pe>90?'N/A':s.pe+'x'}</td></tr>
        <tr><td>P/B Ratio</td><td>${s.pb}x</td></tr>
        <tr><td>Return on Equity</td><td>${(s.roe*100).toFixed(0)}%</td></tr>
        <tr><td>Net Margin</td><td>${(s.mg*100).toFixed(1)}%</td></tr>
        <tr><td>Revenue Growth (YoY)</td><td>${(s.rg*100).toFixed(0)}%</td></tr>
        <tr><td>Debt / Equity</td><td>${s.de.toFixed(2)}x</td></tr>
        <tr><td>Dividend Yield</td><td>${(s.dy*100).toFixed(1)}%</td></tr>
        <tr><td>Beta</td><td>${s.beta.toFixed(2)}${r&&r.betaNote?`<span style="font-size:10px;color:var(--text3);margin-left:6px">⚠ ${esc(r.betaNote)}</span>`:''}</td></tr>
        <tr><td>12M Price Return</td><td style="color:${s.m12>0?'var(--green)':'var(--red)'}">${s.m12>0?'+':''}${(s.m12*100).toFixed(0)}%</td></tr>
        <tr><td>3M Price Return</td><td style="color:${s.m3>0?'var(--green)':'var(--red)'}">${s.m3>0?'+':''}${(s.m3*100).toFixed(0)}%</td></tr>
        <tr><td>Institutional Ownership</td><td style="color:${r&&r.instPct!=null?(r.instPct>70?'var(--green)':r.instPct<30?'var(--red)':'var(--text2)'):'var(--text3)'}">${r&&r.instPct!=null?r.instPct+'%':'—'}</td></tr>
        <tr><td>Insider Ownership</td><td style="color:${r&&r.insiderPct!=null&&r.insiderPct>10?'var(--green)':'var(--text2)'}">${r&&r.insiderPct!=null?r.insiderPct+'%':'—'}</td></tr>
        <tr><td>Sector</td><td>${esc(s.s)}</td></tr>
        <tr><td>Market</td><td>${s.m}</td></tr>
      </table>
    </div>`;

  document.getElementById('detailPanel').classList.add('open');
}

function buildDetailRationale(s){
  const r=[];
  if(s.pe<12)        r.push({t:'pos',i:'✓',txt:`Deep value P/E of ${s.pe}x — materially below market average, suggesting the market is underpricing earnings power.`});
  else if(s.pe<22)   r.push({t:'pos',i:'✓',txt:`Reasonable P/E of ${s.pe}x — fair pricing for a quality business without heroic growth assumptions.`});
  else if(s.pe>50)   r.push({t:'neu',i:'△',txt:`Demanding P/E of ${s.pe}x — the multiple requires sustained high growth; any deceleration risks a sharp re-rating.`});
  if(s.roe>0.30)     r.push({t:'pos',i:'✓',txt:`Exceptional ROE of ${(s.roe*100).toFixed(0)}% — a hallmark of durable competitive advantage and disciplined capital allocation.`});
  else if(s.roe>0.12)r.push({t:'pos',i:'✓',txt:`Solid ROE of ${(s.roe*100).toFixed(0)}% — earns above cost of equity, creating genuine shareholder value.`});
  else               r.push({t:'neu',i:'△',txt:`Modest ROE of ${(s.roe*100).toFixed(0)}% — borderline value creation at current cost of capital.`});
  if(s.mg>0.25)      r.push({t:'pos',i:'✓',txt:`Exceptional net margin of ${(s.mg*100).toFixed(1)}% — strong pricing power and highly scalable business economics.`});
  else if(s.mg>0.10) r.push({t:'pos',i:'✓',txt:`Healthy margin of ${(s.mg*100).toFixed(1)}% — profitable and operationally resilient to moderate cost headwinds.`});
  else               r.push({t:'neu',i:'△',txt:`Thin margin of ${(s.mg*100).toFixed(1)}% — lean operations but vulnerable to revenue shortfalls.`});
  if(s.de<0.4)       r.push({t:'pos',i:'✓',txt:`Conservative balance sheet at ${s.de.toFixed(2)}x D/E — financial fortress providing flexibility through cycles.`});
  else if(s.de>2.0)  r.push({t:'neg',i:'⚠',txt:`Elevated leverage at ${s.de.toFixed(2)}x D/E — materially higher interest rate sensitivity and refinancing risk.`});
  if(s.rg>0.20)      r.push({t:'pos',i:'✓',txt:`Revenue compounding at ${(s.rg*100).toFixed(0)}% annually — rare growth trajectory creating a long value creation runway.`});
  else if(s.rg<0.0)  r.push({t:'neg',i:'⚠',txt:`Declining revenues (${(s.rg*100).toFixed(0)}%) — structural headwinds or competitive pressure limiting top-line growth.`});
  if(s.m12>0.40)     r.push({t:'pos',i:'✓',txt:`Strong 12M momentum of +${(s.m12*100).toFixed(0)}% — price action confirming fundamentals with visible institutional accumulation.`});
  else if(s.m12<-0.08)r.push({t:'neg',i:'⚠',txt:`Negative 12M trend (${(s.m12*100).toFixed(0)}%) — market expressing concern; requires due diligence before entry.`});
  if(s.beta<0.70)    r.push({t:'pos',i:'✓',txt:`Defensive beta of ${s.beta.toFixed(2)} — genuine portfolio ballast during risk-off episodes and corrections.`});
  else if(s.beta>1.6)r.push({t:'neu',i:'△',txt:`High beta of ${s.beta.toFixed(2)} — amplifies both gains and losses; disciplined position sizing is critical.`});
  if(s.dy>0.04)      r.push({t:'pos',i:'✓',txt:`High yield of ${(s.dy*100).toFixed(1)}% — delivers total return floor and signals management confidence in cash generation.`});
  return r.slice(0,6);
}

function buildPeriodExamples(examples){
  if(!examples||examples.length===0)return'';
  const periods=examples.map(p=>{
    const hrColor=p.hit_rate_pct>=60?'var(--green)':p.hit_rate_pct>=40?'var(--amber)':'var(--red)';
    const rows=p.stocks.map(s=>{
      const scoreColor=s.model_score>=50?'var(--green)':'var(--text3)';
      const retColor=s.actual_return_pct>=0?'var(--green)':'var(--red)';
      const retStr=(s.actual_return_pct>0?'+':'')+s.actual_return_pct+'%';
      const correctMark=s.correct?'<span class="bt-correct">✓</span>':'<span class="bt-wrong">✗</span>';
      return`<tr><td style="font-family:'JetBrains Mono',monospace;font-weight:600">${s.ticker}</td><td style="color:${scoreColor}">${s.model_score}%</td><td>${s.predicted_top?'Top half':'Bottom half'}</td><td style="color:${retColor}">${retStr}</td><td>${s.actual_top?'Top half':'Bottom half'}</td><td style="text-align:center">${correctMark}</td></tr>`;
    }).join('');
    return`<div class="bt-period">
      <div class="bt-period-header">
        <span class="bt-period-label">${p.date_range||'Period '+p.period} · ${p.n_stocks} stocks</span>
        <span class="bt-period-hr" style="color:${hrColor}">${p.hit_rate_pct}% correct</span>
      </div>
      <table class="bt-period-table">
        <thead><tr><th>Ticker</th><th>Model Score</th><th>Predicted</th><th>Actual Return</th><th>Actual Rank</th><th>✓/✗</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
  }).join('');
  return`<div class="bt-examples"><div class="bt-examples-title">Out-of-sample prediction examples (holdout periods)</div>${periods}</div>`;
}

async function runBacktest(){
  if(!await checkBackend()){document.getElementById('btResults').innerHTML='<div class="bt-offline">Backtest requires the Python backend to be running. Open this page via the backend server, not as a local file.</div>';return;}
  if(!runResults||!runResults.scored){alert('Please run an analysis first.');return;}
  const btn=document.getElementById('btBtn');
  btn.disabled=true;btn.textContent='RUNNING...';
  document.getElementById('btResults').innerHTML='<div class="bt-loading">Running walk-forward backtest — fetching historical returns...</div>';
  const tickers=runResults.scored.map(s=>s.t);
  const resp=await callBackend('/backtest',{tickers,profile:runResults.profileKey,lookback_years:+document.getElementById('sp-lbSlider').value,forward_months:12});
  const data=resp.ok?resp.data:null;
  btn.disabled=false;btn.textContent='RUN BACKTEST ↗';
  if(!data||data.error){document.getElementById('btResults').innerHTML=`<div class="bt-error">${data&&data.error?data.error:'Backtest failed — check backend connection.'}</div>`;return;}
  const m=data.metrics;
  if(!m||m.n_periods===0){document.getElementById('btResults').innerHTML='<div class="bt-error">Insufficient price history for backtest. Try adding more tickers or increasing the lookback period.</div>';return;}
  function metricColor(val,good,warn){return val>=good?'var(--green)':val>=warn?'var(--amber)':'var(--red)';}
  document.getElementById('btResults').innerHTML=`
    <div style="font-size:10px;color:var(--text3);font-family:'JetBrains Mono',monospace;margin-bottom:10px;letter-spacing:0.05em">${m.n_periods} test periods · ${data.backtest_period} lookback · ${data.forward_months}M forward window · ${tickers.length} tickers</div>
    <div class="bt-grid">
      <div class="bt-metric"><div class="bt-lbl">Group Hit Rate</div><div class="bt-val" style="color:${metricColor(m.hit_rate,55,45)}">${m.hit_rate!=null?m.hit_rate+'%':'—'}</div><div class="bt-interp">Top half beat bottom half</div></div>
      <div class="bt-metric"><div class="bt-lbl">Info. Coeff.</div><div class="bt-val" style="color:${metricColor(m.ic_mean,0.10,0.05)}">${m.ic_mean!=null?m.ic_mean.toFixed(3):'—'}</div><div class="bt-interp">${m.interpretation?.ic||''}</div></div>
      <div class="bt-metric"><div class="bt-lbl">IC IR</div><div class="bt-val" style="color:${metricColor(m.icir,0.5,0.2)}">${m.icir!=null?m.icir.toFixed(2):'—'}</div><div class="bt-interp">IC consistency</div></div>
      <div class="bt-metric"><div class="bt-lbl">Sharpe Ratio</div><div class="bt-val" style="color:${metricColor(m.sharpe_ratio,1,0)}">${m.sharpe_ratio!=null?m.sharpe_ratio.toFixed(2):'—'}</div><div class="bt-interp">${m.interpretation?.sharpe||''}</div></div>
      <div class="bt-metric"><div class="bt-lbl">Ann. Alpha (net)</div><div class="bt-val" style="color:${(m.ann_alpha_net??m.ann_alpha)>0?'var(--green)':'var(--red)'}">${(m.ann_alpha_net??m.ann_alpha)!=null?(((m.ann_alpha_net??m.ann_alpha)>0?'+':'')+(m.ann_alpha_net??m.ann_alpha)+'%'):'—'}</div><div class="bt-interp">gross ${m.ann_alpha!=null?((m.ann_alpha>0?'+':'')+m.ann_alpha+'%'):'—'} · ${m.cost_bps??30}bps costs</div></div>
      <div class="bt-metric"><div class="bt-lbl">Max Drawdown</div><div class="bt-val" style="color:${metricColor(m.max_drawdown,-10,-20)}">${m.max_drawdown!=null?m.max_drawdown+'%':'—'}</div><div class="bt-interp">${m.interpretation?.drawdown||''}</div></div>
      <div class="bt-metric"><div class="bt-lbl">Calmar Ratio</div><div class="bt-val" style="color:${metricColor(m.calmar_ratio,1,0.5)}">${m.calmar_ratio!=null?m.calmar_ratio.toFixed(2):'—'}</div><div class="bt-interp">Return / drawdown</div></div>
      <div class="bt-metric"><div class="bt-lbl">Beta</div><div class="bt-val">${m.beta_vs_benchmark!=null?m.beta_vs_benchmark.toFixed(2):'—'}</div><div class="bt-interp">vs benchmark</div></div>
      <div class="bt-metric"><div class="bt-lbl">Ann. Portfolio (net)</div><div class="bt-val" style="color:${(m.ann_portfolio_net??m.ann_portfolio_return)>0?'var(--green)':'var(--red)'}">${(m.ann_portfolio_net??m.ann_portfolio_return)!=null?(((m.ann_portfolio_net??m.ann_portfolio_return)>0?'+':'')+(m.ann_portfolio_net??m.ann_portfolio_return)+'%'):'—'}</div><div class="bt-interp">gross ${m.ann_portfolio_return!=null?((m.ann_portfolio_return>0?'+':'')+m.ann_portfolio_return+'%'):'—'} · top-quintile</div></div>
    </div>
    ${buildPeriodExamples(m.training?.period_examples)}`;
}

document.getElementById('dpClose').onclick=()=>document.getElementById('detailPanel').classList.remove('open');
document.getElementById('runBtn').onclick=runAnalysis;
initStockList();

/* ═══════════════════ PUBLIC INSIGHTS HERO ═══════════════════ */
// Login-free trust layer: fetches the daily snapshot from /api/insights and
// renders the verified track record + balanced bull/bear theses on the landing
// page. Fails silently to the original welcome screen if unavailable.
async function loadInsights(){
  let data;
  try{
    const r=await fetch('/api/insights');
    if(!r.ok)return;
    data=await r.json();
  }catch(e){return;}
  if(!data||data.status==='warming'||!Array.isArray(data.theses)||!data.theses.length)return;
  renderInsightsHero(data);
}

function _insMetricColor(good){return good?'var(--green)':'var(--text1)';}

function renderInsightsHero(d){
  const el=document.getElementById('insightsHero');
  if(!el)return;
  const tr=d.track_record||{};
  const interp=tr.interpretation||{};
  const hr=tr.hit_rate;

  const head=hr!=null
    ? `The model's top-ranked stocks beat the market <em>${hr}%</em> of the time`
    : `How the model reads the market today`;
  const sub=tr.n_periods
    ? `Walk-forward backtest · ${tr.n_periods} periods · tested on data it never trained on`
    : `Updated daily · ${d.universe_size} stocks across US, UK &amp; Ireland`;

  const metrics=[
    {lbl:'OOS Hit Rate', val:hr!=null?hr+'%':'—', note:interp.hit_rate||'beat the universe', good:hr!=null&&hr>55},
    {lbl:tr.ic_is_holdout?'Holdout IC':'Info Coeff.', val:tr.ic!=null?tr.ic.toFixed(3):'—', note:interp.ic||'scores vs returns', good:tr.ic!=null&&tr.ic>0.05},
    {lbl:'Ann. Alpha', val:tr.ann_alpha!=null?((tr.ann_alpha>0?'+':'')+tr.ann_alpha+'%'):'—', note:interp.alpha||'vs equal-weight', good:tr.ann_alpha!=null&&tr.ann_alpha>0},
    {lbl:'Sharpe', val:tr.sharpe!=null?tr.sharpe.toFixed(2):'—', note:interp.sharpe||'risk-adjusted', good:tr.sharpe!=null&&tr.sharpe>1},
    {lbl:'Max Drawdown', val:tr.max_drawdown!=null?tr.max_drawdown+'%':'—', note:interp.drawdown||'peak-to-trough', good:tr.max_drawdown!=null&&tr.max_drawdown>-20},
  ];
  const metricsHtml=metrics.map(m=>`<div class="ins-metric">
      <div class="ins-metric-lbl">${m.lbl}</div>
      <div class="ins-metric-val" style="color:${_insMetricColor(m.good)}">${m.val}</div>
      <div class="ins-metric-note">${m.note}</div>
    </div>`).join('');

  const buys=d.theses.filter(t=>t.signal==='STRONG BUY'||t.signal==='BUY').slice(0,4);
  const sells=d.theses.filter(t=>t.signal==='SELL').slice(-2);
  let cards=[...buys,...sells];
  if(!cards.length)cards=d.theses.slice(0,6);
  const cardsHtml=cards.map(thesisCard).join('');

  const trackBlock=hr!=null||tr.ic!=null
    ? `<div class="ins-proof-sub">${sub}</div><div class="ins-metrics">${metricsHtml}</div>`
    : '';

  el.innerHTML=`
    <div class="ins-proof-head">${head}</div>
    ${trackBlock}
    <div class="ins-sec-title">Today's calls — the bull case and the bear case for each</div>
    <div class="ins-grid">${cardsHtml}</div>
    <div class="ins-cta">
      <div class="ins-cta-row">
        <button class="ins-cta-btn primary" onclick="document.getElementById('runBtn').click()">RUN THE MODEL ON THESE ↗</button>
        <button class="ins-cta-btn ghost" onclick="Auth.openModal();Auth.showTab('signup')">CREATE FREE ACCOUNT</button>
      </div>
      <div class="ins-cta-note">Free account unlocks custom watchlists, deep per-stock breakdowns &amp; saved analyses.</div>
    </div>
    <div class="ins-divider">OR BUILD YOUR OWN BELOW</div>`;
  el.style.display='flex';
}

function thesisCard(t){
  const isBuy=t.signal&&t.signal.indexOf('BUY')>=0;
  const sigColor=isBuy?'var(--green)':(t.signal==='SELL'?'var(--red)':'var(--amber)');
  const bull=(t.bull&&t.bull[0])?t.bull[0]:'Balanced fundamentals across factors.';
  const bear=(t.bear&&t.bear[0])?t.bear[0]:'No major red flags in the current data.';
  const conf=t.confidence_label?t.confidence_label.toLowerCase():'';
  const hr=t.band_hit_rate!=null
    ? `Stocks rated this confidently beat the market <strong>${t.band_hit_rate}%</strong> of the time`
    : `${t.confidence_label||''} confidence`;
  return `<div class="ins-card">
    <div class="ins-card-top">
      <div><div class="ins-tk">${t.ticker}</div><div class="ins-nm">${t.name||''}</div></div>
      <div class="ins-sig" style="color:${sigColor};border-color:${sigColor}">${t.signal}</div>
    </div>
    <div class="ins-bb"><span class="ins-bb-tag bull">BULL</span><span>${bull}</span></div>
    <div class="ins-bb"><span class="ins-bb-tag bear">BEAR</span><span>${bear}</span></div>
    <div class="ins-hr">${hr}</div>
  </div>`;
}

loadInsights();

/* ═══════════════════ FETCH INTERCEPTOR ═══════════════════ */
// Wraps window.fetch to handle two global error conditions without changing
// any existing call sites:
//
//   401 — Session expired or revoked while the user was active.
//          If we believed the user was logged in, silently reset auth state
//          and re-open the login modal with an explanatory message.
//
//   429 — Rate limit hit. Show a brief toast so the user knows to slow down.
//
// The interceptor only reads response.status — never the body — so callers
// can still read the body themselves without consuming a one-shot stream.
(function _installFetchInterceptor() {
  const _orig = window.fetch.bind(window);
  window.fetch = async function(...args) {
    const response = await _orig(...args);

    if (response.status === 401) {
      // Only act if Auth believes a user is currently logged in.
      // A 401 on /auth/me during init() is expected when not logged in.
      if (typeof Auth !== 'undefined' && Auth.isLoggedIn()) {
        Auth.handleSessionExpiry();
      }
    }

    if (response.status === 429) {
      _showToast('Too many requests — please wait a moment before trying again.', 'warn');
    }

    return response;   // always return the original response unchanged
  };
})();

// ── Toast helper (used by interceptor + any module that needs it) ──────────
function _showToast(msg, type = 'info') {
  const existing = document.getElementById('appToast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.id        = 'appToast';
  toast.className = 'app-toast app-toast-' + type;
  toast.textContent = msg;
  document.body.appendChild(toast);

  // Animate in
  requestAnimationFrame(() => toast.classList.add('app-toast-visible'));

  // Auto-dismiss after 4s
  setTimeout(() => {
    toast.classList.remove('app-toast-visible');
    setTimeout(() => toast.remove(), 350);
  }, 4000);
}

/* ═══════════════════ AUTH MODULE ═══════════════════ */
const Auth = (() => {
  let _user = null;  // { id, email, display_name, created_at } or null

  // ── private helpers ────────────────────────────────────────────────────

  async function _apiPost(path, body, method = 'POST') {
    const r = await fetch(path, {
      method,
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify(body),
    });
    const data = await r.json().catch(() => ({}));
    return { ok: r.ok, status: r.status, data };
  }

  async function _apiGet(path) {
    const r = await fetch(path, { credentials: 'include' });
    const data = await r.json().catch(() => ({}));
    return { ok: r.ok, status: r.status, data };
  }

  function _setUser(user) {
    const wasNull = _user === null;
    _user = user;
    _updateHeaderBtn();
    // Notify Saves module (may not exist yet on first parse, so guard)
    if (typeof Saves !== 'undefined') {
      if (user) Saves.onLogin();
      else Saves.onLogout();
    }
  }

  function _updateHeaderBtn() {
    const btn = document.getElementById('authBtn');
    if (!btn) return;
    if (_user) {
      const name = _user.display_name || _user.email.split('@')[0];
      btn.textContent = name.toUpperCase();
      btn.classList.add('auth-logged-in');
    } else {
      btn.textContent = 'LOG IN';
      btn.classList.remove('auth-logged-in');
    }
  }

  function _setError(id, msg) {
    const el = document.getElementById(id);
    if (el) el.textContent = msg || '';
  }

  function _setLoading(btnId, loading) {
    const btn = document.getElementById(btnId);
    if (!btn) return;
    btn.disabled = loading;
    btn.textContent = loading
      ? (btnId === 'loginSubmit' ? 'LOGGING IN…' : 'CREATING…')
      : (btnId === 'loginSubmit' ? 'LOG IN' : 'CREATE ACCOUNT');
  }

  // ── tab / modal control ────────────────────────────────────────────────

  function showTab(tab) {
    ['login','signup'].forEach(t => {
      document.getElementById('tab'   + t[0].toUpperCase() + t.slice(1)).classList.toggle('active', t === tab);
      document.getElementById('panel' + t[0].toUpperCase() + t.slice(1)).classList.toggle('active', t === tab);
    });
    _setError('loginError', '');
    _setError('signupError', '');
  }

  function _showAccountPanel() {
    document.getElementById('authTabs').style.display = 'none';
    ['Login','Signup'].forEach(p => document.getElementById('panel'+p).classList.remove('active'));
    document.getElementById('panelAccount').classList.add('active');
    if (_user) {
      const initials = (_user.display_name || _user.email).slice(0, 1).toUpperCase();
      document.getElementById('acctAvatar').textContent    = initials;
      document.getElementById('acctName').textContent      = _user.display_name || '—';
      document.getElementById('acctEmail').textContent     = _user.email;
      document.getElementById('acctNameInput').value       = _user.display_name || '';
      document.getElementById('acctNameMsg').textContent   = '';
      document.getElementById('acctNameMsg').className     = 'acct-name-msg';

      // Collapse password form on every open
      const pwdForm = document.getElementById('acctPwdForm');
      if (pwdForm) {
        pwdForm.classList.remove('open');
        document.getElementById('acctPwdToggle').textContent = 'CHANGE PASSWORD ↓';
        ['acctPwdCurrent','acctPwdNew','acctPwdConfirm'].forEach(id => {
          document.getElementById(id).value = '';
        });
        document.getElementById('acctPwdMsg').textContent = '';
      }

      // Collapse delete form on every open
      const delForm = document.getElementById('acctDeleteForm');
      if (delForm) {
        delForm.classList.remove('open');
        const tog = document.getElementById('acctDeleteToggle');
        tog.textContent  = 'DELETE ACCOUNT';
        tog.style.borderColor = '';
        tog.style.color       = '';
        document.getElementById('acctDeletePwd').value       = '';
        document.getElementById('acctDeleteMsg').textContent = '';
      }

      const daysSince = Math.floor((Date.now() - new Date(_user.created_at)) / 86400000);
      document.getElementById('acctDaysSince').textContent = daysSince;
    }
    // Load live stats + activity in the background
    _loadAccountData();
  }

  async function _loadAccountData() {
    // Saved analyses count
    try {
      const r = await _apiGet('/user/analyses');
      if (r.ok) {
        document.getElementById('acctSavedCount').textContent = r.data.length;
      }
    } catch { /* silent */ }

    // Activity feed
    try {
      const r = await _apiGet('/user/history?limit=20');
      if (r.ok && Array.isArray(r.data)) {
        document.getElementById('acctEventCount').textContent = r.data.length < 20 ? r.data.length : '20+';
        _renderActivity(r.data.slice(0, 6));
      }
    } catch { /* silent */ }

    // Active sessions
    try {
      const r = await _apiGet('/user/sessions');
      if (r.ok && Array.isArray(r.data)) _renderSessions(r.data);
    } catch { /* silent */ }
  }

  function _renderSessions(sessions) {
    const el = document.getElementById('acctSessions');
    if (!el) return;
    if (!sessions.length) {
      el.innerHTML = '<div class="acct-activity-empty">No active sessions found.</div>';
      return;
    }
    const _rel = iso => {
      const diff = Date.now() - new Date(iso);
      const mins = Math.floor(diff / 60000);
      if (mins < 1)  return 'just now';
      if (mins < 60) return mins + 'm ago';
      const hrs = Math.floor(mins / 60);
      if (hrs < 24)  return hrs + 'h ago';
      return Math.floor(hrs / 24) + 'd ago';
    };
    const _ua = ua => {
      if (!ua) return 'Unknown device';
      if (/mobile|android/i.test(ua)) return 'Mobile browser';
      if (/chrome/i.test(ua))  return 'Chrome';
      if (/firefox/i.test(ua)) return 'Firefox';
      if (/safari/i.test(ua))  return 'Safari';
      return 'Browser';
    };
    el.innerHTML = sessions.map(s => `
      <div class="acct-session" id="sess-${s.id}">
        <div class="acct-session-info">
          <div class="acct-session-ip">${s.ip_address || 'Unknown IP'}</div>
          <div class="acct-session-ua">${_ua(s.user_agent)}</div>
        </div>
        <div class="acct-session-time">${_rel(s.created_at)}</div>
        <button class="acct-session-revoke" onclick="Auth.revokeSession('${s.id}',this)">REVOKE</button>
      </div>`).join('');
  }

  async function revokeSession(sessionId, btn) {
    btn.disabled = true;
    btn.textContent = '…';
    const r = await fetch('/user/sessions/' + sessionId, { method: 'DELETE', credentials: 'include' });
    const ok = r.ok;
    if (ok) {
      const row = document.getElementById('sess-' + sessionId);
      if (row) row.remove();
      const el = document.getElementById('acctSessions');
      if (el && !el.querySelector('.acct-session')) {
        el.innerHTML = '<div class="acct-activity-empty">No active sessions found.</div>';
      }
      _showToast('Session revoked.', 'ok');
    } else {
      btn.disabled = false;
      btn.textContent = 'REVOKE';
      _showToast('Could not revoke session.', 'err');
    }
  }

  function _renderActivity(events) {
    const el = document.getElementById('acctActivity');
    if (!el) return;
    if (!events.length) {
      el.innerHTML = '<div class="acct-activity-empty">No activity yet.</div>';
      return;
    }
    const _dotClass = type => {
      if (type === 'login' || type === 'signup') return type;
      if (type.includes('analys') || type.includes('backtest')) return 'analysis';
      return '';
    };
    const _label = type => type.replace(/_/g, ' ');
    const _rel = iso => {
      const diff = Date.now() - new Date(iso);
      const mins = Math.floor(diff / 60000);
      if (mins < 1)   return 'just now';
      if (mins < 60)  return mins + 'm ago';
      const hrs = Math.floor(mins / 60);
      if (hrs < 24)   return hrs + 'h ago';
      return Math.floor(hrs / 24) + 'd ago';
    };
    el.innerHTML = events.map(e => `
      <div class="acct-event">
        <div class="acct-event-dot ${_dotClass(e.event_type)}"></div>
        <div class="acct-event-type">${_label(e.event_type)}</div>
        <div class="acct-event-time">${_rel(e.created_at)}</div>
      </div>
    `).join('');
  }

  function _showAuthPanels() {
    document.getElementById('authTabs').style.display = '';
    document.getElementById('panelAccount').classList.remove('active');
    showTab('login');
  }

  function openModal() {
    const overlay = document.getElementById('authOverlay');
    overlay.classList.add('open');
    document.body.style.overflow = 'hidden';
    if (_user) {
      _showAccountPanel();
    } else {
      _showAuthPanels();
      setTimeout(() => document.getElementById('loginEmail').focus(), 80);
    }
  }

  function closeModal() {
    document.getElementById('authOverlay').classList.remove('open');
    document.body.style.overflow = '';
  }

  function handleOverlayClick(e) {
    if (e.target === document.getElementById('authOverlay')) closeModal();
  }

  // ── API calls ──────────────────────────────────────────────────────────

  async function login() {
    _setError('loginError', '');
    const email    = document.getElementById('loginEmail').value.trim();
    const password = document.getElementById('loginPassword').value;
    const remember = document.getElementById('loginRemember').checked;
    if (!email || !password) { _setError('loginError', 'Please fill in all fields.'); return; }
    _setLoading('loginSubmit', true);
    const { ok, data } = await _apiPost('/auth/login', { email, password, remember_me: remember });
    _setLoading('loginSubmit', false);
    if (ok && data.user) {
      _setUser(data.user);
      closeModal();
    } else {
      _setError('loginError', data.detail || 'Login failed. Please try again.');
    }
  }

  async function signup() {
    _setError('signupError', '');
    const display_name = document.getElementById('signupName').value.trim() || null;
    const email        = document.getElementById('signupEmail').value.trim();
    const password     = document.getElementById('signupPassword').value;
    if (!email || !password) { _setError('signupError', 'Email and password are required.'); return; }
    _setLoading('signupSubmit', true);
    const { ok, data } = await _apiPost('/auth/signup', { email, password, display_name });
    _setLoading('signupSubmit', false);
    if (ok && data.user) {
      _setUser(data.user);
      closeModal();
    } else {
      _setError('signupError', data.detail || 'Sign up failed. Please try again.');
    }
  }

  async function logout() {
    await _apiPost('/auth/logout', {});
    _setUser(null);
    closeModal();
  }

  async function init() {
    // Silently check if the user already has a valid session cookie.
    // Saves module is defined after Auth, so notify it after the event loop tick.
    try {
      const { ok, data } = await _apiGet('/auth/me');
      if (ok && data.user) {
        _user = data.user;
        _updateHeaderBtn();
        // Saves is now defined (async boundary guarantees script has fully parsed)
        if (typeof Saves !== 'undefined') Saves.onLogin();
      }
    } catch {
      // No database or network issue — silent fail, app still works
    }
  }

  function togglePwdForm() {
    const form   = document.getElementById('acctPwdForm');
    const toggle = document.getElementById('acctPwdToggle');
    const open   = form.classList.toggle('open');
    toggle.textContent = open ? 'CHANGE PASSWORD ↑' : 'CHANGE PASSWORD ↓';
    if (open) {
      document.getElementById('acctPwdCurrent').focus();
    } else {
      // Clear fields when collapsing
      ['acctPwdCurrent','acctPwdNew','acctPwdConfirm'].forEach(id => {
        document.getElementById(id).value = '';
      });
      document.getElementById('acctPwdMsg').textContent = '';
      document.getElementById('acctPwdMsg').className   = 'acct-pwd-msg';
    }
  }

  async function changePassword() {
    const current  = document.getElementById('acctPwdCurrent').value;
    const newPwd   = document.getElementById('acctPwdNew').value;
    const confirm  = document.getElementById('acctPwdConfirm').value;
    const msgEl    = document.getElementById('acctPwdMsg');
    const btn      = document.getElementById('acctPwdSubmit');

    const _msg = (text, type = '') => {
      msgEl.textContent = text;
      msgEl.className   = 'acct-pwd-msg' + (type ? ' ' + type : '');
    };

    if (!current || !newPwd || !confirm) { _msg('All fields are required.', 'err'); return; }
    if (newPwd.length < 8)               { _msg('New password must be at least 8 characters.', 'err'); return; }
    if (newPwd !== confirm)              { _msg('Passwords do not match.', 'err'); return; }

    btn.disabled = true;
    _msg('Updating…');

    const { ok, data } = await _apiPost('/auth/change-password', {
      current_password: current,
      new_password:     newPwd,
    });

    btn.disabled = false;

    if (ok) {
      _msg(data.message || 'Password updated successfully.', 'ok');
      ['acctPwdCurrent','acctPwdNew','acctPwdConfirm'].forEach(id => {
        document.getElementById(id).value = '';
      });
      // Collapse the form after a short delay
      setTimeout(() => togglePwdForm(), 2200);
    } else {
      _msg(data.detail || 'Could not update password.', 'err');
    }
  }

  async function saveDisplayName() {
    const input = document.getElementById('acctNameInput');
    const msgEl = document.getElementById('acctNameMsg');
    const btn   = document.getElementById('acctNameSave');
    const name  = input.value.trim();

    btn.disabled = true;
    msgEl.className = 'acct-name-msg';
    msgEl.textContent = 'Saving…';

    const { ok, data } = await _apiPost('/auth/me', { display_name: name || null },  'PATCH');
    btn.disabled = false;

    if (ok && data.user) {
      _user = data.user;
      _updateHeaderBtn();
      const initials = (data.user.display_name || data.user.email).slice(0, 1).toUpperCase();
      document.getElementById('acctAvatar').textContent = initials;
      document.getElementById('acctName').textContent   = data.user.display_name || '—';
      msgEl.className   = 'acct-name-msg ok';
      msgEl.textContent = 'Saved!';
      setTimeout(() => { if (msgEl.textContent === 'Saved!') msgEl.textContent = ''; }, 2500);
    } else {
      msgEl.className   = 'acct-name-msg err';
      msgEl.textContent = data.detail || 'Could not save.';
    }
  }

  function toggleDeleteForm() {
    const form   = document.getElementById('acctDeleteForm');
    const toggle = document.getElementById('acctDeleteToggle');
    const open   = form.classList.toggle('open');
    toggle.textContent = open ? 'CANCEL' : 'DELETE ACCOUNT';
    toggle.style.borderColor = open ? 'var(--border2)' : '';
    toggle.style.color       = open ? 'var(--text3)'   : '';
    if (open) {
      document.getElementById('acctDeletePwd').focus();
    } else {
      document.getElementById('acctDeletePwd').value      = '';
      document.getElementById('acctDeleteMsg').textContent = '';
    }
  }

  async function deleteAccount() {
    const pwd   = document.getElementById('acctDeletePwd').value;
    const msgEl = document.getElementById('acctDeleteMsg');
    const btn   = document.getElementById('acctDeleteSubmit');

    if (!pwd) { msgEl.textContent = 'Enter your password to confirm.'; return; }

    btn.disabled     = true;
    msgEl.textContent = 'Deleting…';

    const r = await fetch('/auth/me', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ password: pwd }),
    });
    const data = await r.json().catch(() => ({}));

    if (r.ok) {
      // Wipe local state, close modal, reset UI
      _setUser(null);
      closeModal();
    } else {
      btn.disabled      = false;
      msgEl.textContent = data.detail || 'Deletion failed. Check your password.';
    }
  }

  function isLoggedIn() {
    return _user !== null;
  }

  function handleSessionExpiry() {
    _setUser(null);
    // Reopen the modal on the login tab with a contextual message
    _showAuthPanels();
    openModal();
    // Briefly surface an error in the login form so the user knows why
    setTimeout(() => {
      _setError('loginError', 'Your session expired — please log in again.');
    }, 80);
  }

  return { init, openModal, closeModal, handleOverlayClick, showTab, login, signup, logout, saveDisplayName, togglePwdForm, changePassword, toggleDeleteForm, deleteAccount, revokeSession, isLoggedIn, handleSessionExpiry };
})();

Auth.init();

/* ═══════════════════ PREFS MODULE ═══════════════════ */
const Prefs = (() => {

  // ── apply prefs to the sidebar ─────────────────────────────────────────

  function _applyToUI(prefs) {
    if (!prefs) return;

    // Profile selector
    const profileSel = document.getElementById('profileSel');
    if (profileSel && prefs.default_profile) profileSel.value = prefs.default_profile;

    // Risk selector
    const riskSel = document.getElementById('riskSel');
    if (riskSel && prefs.default_risk) riskSel.value = prefs.default_risk;

    // Lookback slider
    if (prefs.lookback_years) {
      const slider = document.getElementById('lbSlider');
      const label  = document.getElementById('lbVal');
      if (slider) { slider.value = prefs.lookback_years; }
      if (label)  { label.textContent = prefs.lookback_years + ' yr'; }
    }

    // Default tickers — replace current selection
    if (Array.isArray(prefs.default_tickers) && prefs.default_tickers.length > 0) {
      selectedTickers = new Set(prefs.default_tickers);
      initStockList();
    }
  }

  // ── fetch and apply on login ───────────────────────────────────────────

  async function loadAndApply() {
    try {
      const r = await fetch('/user/preferences', { credentials: 'include' });
      if (!r.ok) return;
      const prefs = await r.json();
      _applyToUI(prefs);
    } catch {
      // Silent fail — app works fine with defaults
    }
  }

  // ── save current sidebar state as defaults ─────────────────────────────

  async function saveDefaults() {
    const btn = document.getElementById('saveDefaultsBtn');
    if (btn) { btn.classList.add('saving'); btn.textContent = 'SAVING…'; }

    const payload = {
      default_profile: document.getElementById('profileSel')?.value || undefined,
      default_risk:    document.getElementById('riskSel')?.value    || undefined,
      lookback_years:  +(document.getElementById('lbSlider')?.value || 5),
      default_tickers: [...selectedTickers],
    };

    try {
      const r = await fetch('/user/preferences', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(payload),
      });

      if (btn) {
        if (r.ok) {
          btn.textContent = 'DEFAULTS SAVED ✓';
          btn.style.color = 'var(--primary)';
          setTimeout(() => {
            btn.textContent = 'SAVE AS MY DEFAULTS';
            btn.style.color = '';
            btn.classList.remove('saving');
          }, 2000);
        } else {
          btn.textContent = 'SAVE FAILED';
          btn.style.color = '#f87171';
          setTimeout(() => {
            btn.textContent = 'SAVE AS MY DEFAULTS';
            btn.style.color = '';
            btn.classList.remove('saving');
          }, 2000);
        }
      }
    } catch {
      if (btn) { btn.classList.remove('saving'); btn.textContent = 'SAVE AS MY DEFAULTS'; }
    }
  }

  return { loadAndApply, saveDefaults };
})();

/* ═══════════════════ SAVES MODULE ═══════════════════ */
const Saves = (() => {
  // ── private helpers ────────────────────────────────────────────────────

  function _authPanels(visible) {
    document.querySelectorAll('.auth-hidden').forEach(el => {
      el.classList.toggle('auth-hidden', !visible);
    });
  }

  function _setMsg(text, type = '') {
    const el = document.getElementById('saveMsg');
    if (!el) return;
    el.textContent = text;
    el.className = 'save-bar-msg' + (type ? ' ' + type : '');
    if (type === 'ok') setTimeout(() => { if (el.textContent === text) el.textContent = ''; }, 3000);
  }

  function _fmtDate(iso) {
    const d = new Date(iso);
    return d.toLocaleDateString('en-GB', { day:'numeric', month:'short', year:'numeric' });
  }

  // ── auth visibility ────────────────────────────────────────────────────

  function onLogin() {
    // Show all .auth-hidden elements by removing the class
    document.querySelectorAll('.auth-hidden').forEach(el => el.classList.remove('auth-hidden'));
    // Restore savedPanel display — renderResults hides it via inline style
    // so removing the class alone isn't enough when results are not showing
    const resultsVisible = document.getElementById('resultsScreen')?.style.display === 'block';
    const savedPanel = document.getElementById('savedPanel');
    if (savedPanel && !resultsVisible) savedPanel.style.display = '';
    loadList();
    // Apply saved preferences to the sidebar
    Prefs.loadAndApply();
  }

  function onLogout() {
    // Re-hide by re-adding the class
    document.getElementById('saveBar')?.classList.add('auth-hidden');
    document.getElementById('savedPanel')?.classList.add('auth-hidden');
    document.getElementById('saveDefaultsBtn')?.classList.add('auth-hidden');
    document.getElementById('savedList').innerHTML = '<div class="saved-empty">Log in to see your saved analyses.</div>';
  }

  // ── save current analysis ──────────────────────────────────────────────

  async function saveCurrentAnalysis() {
    if (!runResults) { _setMsg('Run an analysis first.', 'err'); return; }
    const nameEl = document.getElementById('saveNameInput');
    const name = nameEl.value.trim();
    if (!name) { _setMsg('Enter a name first.', 'err'); nameEl.focus(); return; }

    const btn = document.getElementById('saveBtn');
    btn.disabled = true;
    _setMsg('Saving…');

    const payload = {
      name,
      profile: runResults.profileKey,
      risk:    runResults.riskKey,
      tickers: runResults.scored.map(s => s.t),
      results: {
        // Store every field renderResults, buildBuyReasons, renderCharts need.
        // Omitting any of these causes broken table rows and missing chart data
        // when the analysis is loaded back from the server.
        scored: runResults.scored.map(s => ({
          t: s.t, n: s.n, s: s.s, m: s.m, p: s.p,
          score: s.score, fundScore: s.fundScore,
          signal: s.signal, buyReasons: s.buyReasons || [],
          ff: s.ff || { value:0, quality:0, investment:0, size:0, momentum:0 },
          pe: s.pe, roe: s.roe, mg: s.mg, de: s.de,
          dy: s.dy, rg: s.rg, m12: s.m12, beta: s.beta, mc: s.mc,
          instPct: s.instPct ?? null, insiderPct: s.insiderPct ?? null,
          isLive: s.isLive,
          analystBuy: s.analystBuy ?? null, analystHold: s.analystHold ?? null,
          analystSell: s.analystSell ?? null, daysToEarnings: s.daysToEarnings ?? null,
        })),
        cvAccuracy:        runResults.cvAccuracy,
        cvIC:              runResults.cvIC,
        featureImportance: runResults.featureImportance,
        backendLive:       runResults.backendLive,
        profileKey:        runResults.profileKey,
        riskKey:           runResults.riskKey,
        folds:             runResults.folds,
        lambda:            runResults.lambda,
      },
    };

    try {
      const r = await fetch('/user/analyses', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(payload),
      });
      if (r.ok) {
        _setMsg('Saved!', 'ok');
        nameEl.value = '';
        loadList();
      } else {
        const d = await r.json().catch(() => ({}));
        _setMsg(d.detail || 'Save failed.', 'err');
      }
    } catch {
      _setMsg('Network error.', 'err');
    } finally {
      btn.disabled = false;
    }
  }

  // ── list ───────────────────────────────────────────────────────────────

  async function loadList() {
    const container = document.getElementById('savedList');
    if (!container) return;
    container.innerHTML = '<div class="saved-empty">Loading…</div>';

    try {
      const r = await fetch('/user/analyses', { credentials: 'include' });
      if (r.status === 401) {
        container.innerHTML = '<div class="saved-empty">Log in to see your saved analyses.</div>';
        return;
      }
      if (!r.ok) { container.innerHTML = '<div class="saved-empty">Could not load saved analyses.</div>'; return; }

      const list = await r.json();
      if (!list.length) {
        container.innerHTML = '<div class="saved-empty">No saved analyses yet. Run an analysis and save it above.</div>';
        return;
      }

      container.innerHTML = list.map(item => `
        <div class="saved-item" data-id="${item.id}">
          <div class="saved-item-info">
            <div class="saved-item-name">${_esc(item.name)}</div>
            <div class="saved-item-meta">${item.profile.toUpperCase()} · ${item.risk.toUpperCase()} · ${item.tickers.length} stocks · ${_fmtDate(item.created_at)}</div>
          </div>
          <button class="saved-item-load" onclick="Saves.loadAnalysis('${item.id}')">LOAD</button>
          <button class="saved-item-del"  onclick="Saves.deleteAnalysis('${item.id}', this)">✕</button>
        </div>
      `).join('');
    } catch {
      container.innerHTML = '<div class="saved-empty">Could not load saved analyses.</div>';
    }
  }

  // ── load one analysis ──────────────────────────────────────────────────

  async function loadAnalysis(id) {
    try {
      const r = await fetch(`/user/analyses/${id}`, { credentials: 'include' });
      if (!r.ok) { _showToast('Could not load this analysis.', 'err'); return; }
      const item = await r.json();
      const res  = item.results || {};

      // All rendering fields were saved in full — pass them straight through.
      // Fall back to safe defaults for any field missing in older saves.
      const R = {
        scored:            (res.scored || []).map(s => ({
          ...s,
          s:          s.s   || 'Unknown',
          m:          s.m   || 'US',
          p:          s.p   || 0,
          ff:         s.ff  || { value:0, quality:0, investment:0, size:0, momentum:0 },
          buyReasons: s.buyReasons || [],
          pe: s.pe ?? null, roe: s.roe ?? null, mg: s.mg ?? null,
          de: s.de ?? null, dy: s.dy ?? null,   rg: s.rg ?? null,
          m12: s.m12 ?? null, beta: s.beta ?? null, mc: s.mc ?? null,
          instPct: s.instPct ?? null, insiderPct: s.insiderPct ?? null,
          analystBuy: s.analystBuy ?? null, analystHold: s.analystHold ?? null,
          analystSell: s.analystSell ?? null, daysToEarnings: s.daysToEarnings ?? null,
        })),
        profileKey:        res.profileKey        || item.profile,
        riskKey:           res.riskKey           || item.risk,
        profile:           PROFILES[item.profile] || PROFILES['quality'],
        useFF5:            true,
        folds:             res.folds  || 5,
        lambda:            res.lambda || 0.10,
        cvAccuracy:        res.cvAccuracy        || null,
        cvIC:              res.cvIC              ?? null,
        featureImportance: res.featureImportance || {},
        backendLive:       false,
      };
      runResults = R;
      renderResults(R);
      _showToast(`Loaded "${item.name}"`, 'ok');
    } catch {
      _showToast('Could not load this analysis.', 'err');
    }
  }

  // ── delete ─────────────────────────────────────────────────────────────

  async function deleteAnalysis(id, btn) {
    // Two-stage: first click arms the button, second click confirms.
    // Reset any other armed buttons first so only one is armed at a time.
    if (!btn.dataset.armed) {
      document.querySelectorAll('.saved-item-del[data-armed]').forEach(b => {
        delete b.dataset.armed;
        b.textContent = '✕';
        b.style.borderColor = '';
        b.style.color = '';
      });
      btn.dataset.armed  = '1';
      btn.textContent    = 'SURE?';
      btn.style.borderColor = '#f87171';
      btn.style.color       = '#f87171';
      // Auto-disarm after 4s
      setTimeout(() => {
        if (btn.dataset.armed) {
          delete btn.dataset.armed;
          btn.textContent    = '✕';
          btn.style.borderColor = '';
          btn.style.color       = '';
        }
      }, 4000);
      return;
    }

    // Second click — confirmed
    delete btn.dataset.armed;
    btn.disabled = true;
    try {
      const r = await fetch(`/user/analyses/${id}`, {
        method: 'DELETE',
        credentials: 'include',
      });
      if (r.ok || r.status === 204) {
        const item = btn.closest('.saved-item');
        item.style.transition = 'opacity 0.2s';
        item.style.opacity    = '0';
        setTimeout(() => {
          item.remove();
          const container = document.getElementById('savedList');
          if (container && !container.querySelector('.saved-item')) {
            container.innerHTML = '<div class="saved-empty">No saved analyses yet. Run an analysis and save it above.</div>';
          }
        }, 200);
      } else {
        btn.disabled = false;
        _showToast('Could not delete this analysis.', 'err');
      }
    } catch {
      btn.disabled = false;
      _showToast('Network error — please try again.', 'err');
    }
  }

  // ── html escape ────────────────────────────────────────────────────────

  function _esc(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  return { onLogin, onLogout, saveCurrentAnalysis, loadList, loadAnalysis, deleteAnalysis };
})();

/* ═══ INITIALISE ═══ */
showPage('landing');
initStockList();