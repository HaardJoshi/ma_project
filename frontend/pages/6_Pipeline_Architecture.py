import json
import streamlit as st
from frontend.utils import load_betweenness_data, setup_page

setup_page("Pipeline Architecture")

# Load betweenness data for Step 2 (GraphSAGE) mapping
betweenness_top10 = load_betweenness_data()

st.markdown("### Internal Engine & Pipeline Architecture")

# At the top of the page, initialise state
if 'pipeline_stage' not in st.session_state:
    st.session_state.pipeline_stage = 0

STAGES = [
    {"id": 0, "label": "Data Ingestion",     "sub": "Bloomberg · EDGAR · SPLC"},
    {"id": 1, "label": "CAR Formulation",    "sub": "Event Study · CAPM"},
    {"id": 2, "label": "GraphSAGE",          "sub": "Supply Chain Graph"},
    {"id": 3, "label": "FinBERT",            "sub": "MD&A Embeddings"},
    {"id": 4, "label": "Multimodal Fusion",  "sub": "249-Feature Vector"},
    {"id": 5, "label": "XGBoost Inference",  "sub": "AUC 0.566 · p=0.038"},
]

col_nav, col_viz = st.columns([1, 2.5])

with col_nav:
    # Read from query params if set
    params = st.query_params
    if "stage" in params:
        st.session_state.pipeline_stage = int(params["stage"])
    
    active = st.session_state.pipeline_stage
    
    # Build stepper HTML
    steps_html = ""
    for s in STAGES:
        is_active = s["id"] == active
        is_done   = s["id"] < active
        dot_color = "#00FFAA" if is_active else ("#555" if not is_done else "#2a6b4f")
        label_color = "#ffffff" if is_active else "#888"
        font_weight = "600" if is_active else "400"
        connector = '<div style="width:2px;height:24px;background:#2a2a2a;margin:0 auto;margin-left:15px;"></div>' if s["id"] < 5 else ""
        
        steps_html += f"""<a href="?stage={s['id']}" target="_self" style="text-decoration:none;">
<div style="display:flex;align-items:flex-start;gap:12px;padding:8px 12px;border-radius:8px;cursor:pointer;background:{'rgba(0,255,170,0.07)' if is_active else 'transparent'};border:{'1px solid rgba(0,255,170,0.3)' if is_active else '1px solid transparent'};margin-bottom:4px;transition:all 0.2s;">
<div style="width:12px;height:12px;border-radius:50%;background:{dot_color};margin-top:4px;flex-shrink:0;{'box-shadow:0 0 8px #00FFAA;' if is_active else ''}"></div>
<div>
<div style="color:{label_color};font-size:13px;font-weight:{font_weight};font-family:monospace;">{s['label']}</div>
<div style="color:#555;font-size:11px;font-family:monospace;">{s['sub']}</div>
</div>
</div>
</a>
{connector}"""
    
    st.markdown(
        f"<div style='padding:8px 0;'>{steps_html}</div>",
        unsafe_allow_html=True
    )
    
    # Mini stats card for active stage
    stage_stats = {
        0: "4,999 deals · 249 features · 28-year span (1995–2023)",
        1: "Event window: [−3, +3] · Estimation: 252 days",
        2: "5,730 nodes · 7,582 edges · GraphSAGE AUC 0.8245",
        3: "1,674 text-coverage deals · 768 → 32 PCA dims",
        4: "67 financial + 32 text + 150 graph = 249 features",
        5: "XGBoost · 300 estimators · AUC 0.566 · p = 0.038",
    }
    st.info(f"**Stage {active + 1}:** {stage_stats[active]}")

stage_idx = st.session_state.pipeline_stage

with col_viz:
    if stage_idx == 0:
        html_str = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
          body { margin: 0; padding: 0; background-color: #0e1117; overflow: hidden; }
          canvas { display: block; width: 100%; height: 380px; }
        </style>
        </head>
        <body>
        <canvas id="stage-canvas"></canvas>
        <script>
        const canvas = document.getElementById('stage-canvas');
        const ctx = canvas.getContext('2d');

        // Set dimensions on init and resize
        function resize() {
            canvas.width = canvas.offsetWidth;
            canvas.height = 380;
        }
        window.addEventListener('resize', resize);
        resize();

        const COLORS = {
          bg: '#0e1117',
          accent: '#00FFAA',
          financial: '#5591c7',
          textual: '#fdab43',
          graph: '#00FFAA',
          text: '#cdccca',
          muted: '#444',
          surface: '#1c1b19'
        };

        let startTime = null;
        const DURATION = 6000;

        const tickers = ["AAPL US", "MSFT US", "NVDA US", "JPM US", "GS US", "V US", "WMT US", "DIS US", "NFLX US", "TSLA US", "META US", "AMZN US"];
        const forms = ["10-K", "8-K", "DEF 14A", "10-Q", "S-4", "SC 13D", "424B2", "13F-HR", "10-K/A", "8-K/A", "SD", "ARS"];

        function draw(progress, ctx, canvas) {
            const w = canvas.width;
            const h = canvas.height;
            const colW = w / 3;
            
            ctx.font = "14px monospace";
            ctx.textAlign = "center";
            
            // 1. Financial Stream (Blue) - scrolling text UPWARD
            ctx.fillStyle = COLORS.financial;
            let offset1 = (progress * h * 1.5) % 30; // 30px row height
            for(let i=0; i<12; i++) {
                let y = h * 0.7 - (i * 30) + offset1;
                if (y > 0 && y < h * 0.7) {
                    ctx.fillText(tickers[i % tickers.length], colW * 0.5, y);
                }
            }
            
            // 2. Textual Stream (Amber) - scrolling text UPWARD
            ctx.fillStyle = COLORS.textual;
            let offset2 = (progress * h * 1.5) % 30;
            for(let i=0; i<12; i++) {
                let y = h * 0.7 - (i * 30) + offset2;
                if (y > 0 && y < h * 0.7) {
                    ctx.fillText(forms[i % forms.length], colW * 1.5, y);
                }
            }
            
            // 3. Graph Stream (Green) - nodes dropping DOWNWARD
            ctx.fillStyle = COLORS.graph;
            ctx.strokeStyle = `rgba(0, 255, 170, 0.4)`;
            ctx.lineWidth = 1.5;
            let dropOffset = (progress * h * 1.5) % 40;
            // Lines
            for(let i=-2; i<12; i++) {
                let y = (i * 40) + dropOffset;
                let nextY = ((i+1) * 40) + dropOffset;
                if (y < h * 0.7 && nextY > 0) {
                    ctx.beginPath();
                    ctx.moveTo(colW * 2.5, Math.max(0, y));
                    ctx.lineTo(colW * 2.5, Math.min(h * 0.7, nextY));
                    ctx.stroke();
                }
            }
            // Circles
            for(let i=-2; i<12; i++) {
                let y = (i * 40) + dropOffset;
                if (y < h * 0.7 && y > 0) {
                    ctx.beginPath();
                    ctx.arc(colW * 2.5, y, 4, 0, Math.PI*2);
                    ctx.fill();
                }
            }
            
            // 4. Feature Matrix Bar
            const barY = h * 0.75;
            const barH = 20;
            const barW = w * 0.8;
            const startX = w * 0.1;
            
            ctx.fillStyle = COLORS.surface;
            ctx.fillRect(startX, barY, barW, barH);
            
            const fillProgress = Math.min(1.0, progress * 1.2); 
            const currentFillW = barW * fillProgress;
            
            ctx.save();
            ctx.beginPath();
            ctx.rect(startX, barY, currentFillW, barH);
            ctx.clip();
            
            ctx.fillStyle = COLORS.financial;
            ctx.fillRect(startX, barY, barW/3, barH);
            ctx.fillStyle = COLORS.textual;
            ctx.fillRect(startX + barW/3, barY, barW/3, barH);
            ctx.fillStyle = COLORS.graph;
            ctx.fillRect(startX + barW*2/3, barY, barW/3, barH);
            ctx.restore();
            
            ctx.fillStyle = COLORS.text;
            ctx.font = "12px monospace";
            ctx.textAlign = "center";
            ctx.fillText("249-Feature Vector", w/2, barY - 10);
            
            // 5. Progress Counter
            const minDeals = 0;
            const maxDeals = 4999;
            const deals = Math.floor(fillProgress * maxDeals);
            ctx.fillStyle = "#ffffff";
            ctx.font = "14px monospace";
            ctx.fillText(`${deals.toLocaleString()} → ${maxDeals.toLocaleString()} deals loaded`, w/2, barY + 45);
        }

        function animate(timestamp) {
            if (!startTime) startTime = timestamp;
            const elapsed = timestamp - startTime;
            const progress = Math.min(1.0, elapsed / DURATION);
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = COLORS.bg;
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            draw(progress, ctx, canvas);
            
            if (progress < 1.0) {
                requestAnimationFrame(animate);
            }
        }

        requestAnimationFrame(animate);
        </script>
        </body>
        </html>
        """
        st.components.v1.html(html_str, height=400)
        
        st.latex(r"\mathbf{X} \in \mathbb{R}^{N \times 249}, \quad N = 4{,}999")
        
        st.info("💡 **Concept:** Three modalities are combined into a single feature matrix. Financial ratios capture what the numbers show. MD&A text captures what management says. Supply chain graph structure captures what the market knows about the firm's position. No single modality is sufficient alone.")
        
        with st.expander("📂 View Source Code"):
            st.code('''def fuse_modalities(df_fin, df_text, df_graph):
    """
    Late fusion of three distinct embedding spaces.
    Total features = 67 (Fin) + 32 (Text PCA) + 150 (GraphSAGE) = 249
    """
    # 1. Align temporal indices
    merged = df_fin.merge(df_text, on=['Deal_ID', 'Announcement_Date'], how='inner')
    merged = merged.merge(df_graph, on='Deal_ID', how='inner')
    
    # 2. Extract feature arrays
    X_fin = merged[[f'fin_{i}' for i in range(67)]].values
    X_txt = merged[[f'txt_pca_{i}' for i in range(32)]].values
    X_gph = merged[[f'sage_{i}' for i in range(150)]].values
    
    # 3. Multimodal Concatenation (Late Fusion)
    X_fused = np.hstack([X_fin, X_txt, X_gph])
    
    return X_fused, merged['CAR_Label'].values
''', language="python")

    elif stage_idx == 1:
        html_str = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
          body { margin: 0; padding: 0; background-color: #0e1117; overflow: hidden; }
          canvas { display: block; width: 100%; height: 380px; }
        </style>
        </head>
        <body>
        <canvas id="stage-canvas"></canvas>
        <script>
        const canvas = document.getElementById('stage-canvas');
        const ctx = canvas.getContext('2d');

        function resize() {
            canvas.width = canvas.offsetWidth;
            canvas.height = 380;
        }
        window.addEventListener('resize', resize);
        resize();

        const COLORS = {
          bg: '#0e1117',
          accent: '#00FFAA',
          financial: '#5591c7',
          textual: '#fdab43',
          graph: '#00FFAA',
          surface: '#1c1b19'
        };

        let startTime = null;
        const DURATION = 10000;

        function draw(progress, ctx, canvas) {
            const w = canvas.width;
            const h = canvas.height;
            const chartH = 260;
            const midY = chartH / 2;
            
            // local progress: 0->1 for each half
            const isPositive = progress < 0.5 && progress < 1.0;
            const t_local = progress >= 1.0 ? 1.0 : (progress % 0.5) * 2;
            
            const phase1 = Math.min(1, Math.max(0, t_local / 0.2));
            const phase2 = Math.min(1, Math.max(0, (t_local - 0.2) / 0.2));
            const phase3 = Math.min(1, Math.max(0, (t_local - 0.4) / 0.2));
            const phase4 = Math.min(1, Math.max(0, (t_local - 0.6) / 0.2));

            const market = [];
            const expected = [];
            const actual = [];

            for(let x=0; x<=w; x++) {
                let t_day = (x / w) * 20 - 10;
                let m = Math.sin(x * 0.02) * 20 + Math.cos(x * 0.04) * 10;
                market.push(midY - m);
                
                let beta = isPositive ? 1.05 : 0.95;
                let expectedY = midY - (m * beta);
                expected.push(expectedY);
                
                let actualY = expectedY - (Math.sin(x*0.15)*4); // baseline noise
                
                if (t_day >= -3 && t_day <= 3) {
                    let effect = Math.cos(t_day * Math.PI / 6) * 35; // bell curve for CAR
                    if (!isPositive) effect = -effect;
                    actualY -= effect;
                }
                actual.push(actualY);
            }

            // 1. Draw Market Benchmark (Grey Dashed)
            ctx.strokeStyle = '#666';
            ctx.setLineDash([5, 5]);
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for(let x=0; x<=w*phase1; x++) {
                if(x===0) ctx.moveTo(x, market[x]);
                else ctx.lineTo(x, market[x]);
            }
            ctx.stroke();
            ctx.setLineDash([]);
            
            // Label Benchmark
            if (phase1 === 1) {
                ctx.fillStyle = '#666';
                ctx.font = "12px monospace";
                ctx.textAlign = "left";
                ctx.fillText("Market (Rₘ)", 10, market[10] + 15);
            }

            // 2. Draw Actual Return (Blue)
            ctx.strokeStyle = COLORS.financial; 
            ctx.lineWidth = 2;
            ctx.beginPath();
            for(let x=0; x<=w*phase1; x++) {
                if(x===0) ctx.moveTo(x, actual[x]);
                else ctx.lineTo(x, actual[x]);
            }
            ctx.stroke();
            
            if (phase1 === 1) {
                ctx.fillStyle = COLORS.financial;
                ctx.fillText("Actual Target Stock (R)", 10, actual[10] - 10);
            }

            // 3. Draw Expected Return (Amber)
            if (phase2 > 0) {
                ctx.strokeStyle = COLORS.textual;
                ctx.lineWidth = 2.5;
                ctx.beginPath();
                for(let x=0; x<=w*phase2; x++) {
                    if(x===0) ctx.moveTo(x, expected[x]);
                    else ctx.lineTo(x, expected[x]);
                }
                ctx.stroke();
                
                if (phase2 === 1) {
                    ctx.fillStyle = COLORS.textual;
                    ctx.fillText("CAPM Expected (R̂)", w - 130, expected[w-1] + 20);
                }
            }

            // 4. Shade Region & Labels
            if (phase3 > 0) {
                let xStart = Math.floor(w * (7/20));
                let xEnd = Math.floor(w * (13/20));
                
                ctx.fillStyle = isPositive ? `rgba(0,255,170,${0.25 * phase3})` : `rgba(255,100,100,${0.25 * phase3})`;
                
                ctx.beginPath();
                ctx.moveTo(xStart, expected[xStart]);
                for(let x=xStart; x<=xEnd; x++) ctx.lineTo(x, expected[x]);
                for(let x=xEnd; x>=xStart; x--) ctx.lineTo(x, actual[x]);
                ctx.closePath();
                ctx.fill();
                
                ctx.strokeStyle = `rgba(255,255,255,${0.4 * phase3})`;
                ctx.setLineDash([4,4]);
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(xStart, 10); ctx.lineTo(xStart, chartH-10);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(xEnd, 10); ctx.lineTo(xEnd, chartH-10);
                ctx.stroke();
                ctx.setLineDash([]);
                
                ctx.fillStyle = `rgba(255,255,255,${phase3})`;
                ctx.textAlign = "center";
                ctx.fillText("−3 days", xStart, chartH - 10);
                ctx.fillText("+3 days", xEnd, chartH - 10);
                
                let peakX = Math.floor(w/2);
                let peakY = (expected[peakX] + actual[peakX])/2;
                ctx.font = "13px monospace";
                ctx.fillText("CAR = Σ(Rₜ − R̂ₜ)", peakX, peakY + 4);
            }

            // 5. Formula Area
            if (phase4 > 0) {
                ctx.fillStyle = COLORS.surface;
                ctx.fillRect(0, chartH, w, h - chartH);
                
                ctx.textAlign = "left";
                ctx.font = "14px monospace";
                
                let fm1 = Math.min(1, phase4 * 3);
                let fm2 = Math.min(1, Math.max(0, (phase4 - 0.33) * 3));
                let fm3 = Math.min(1, Math.max(0, (phase4 - 0.66) * 3));
                
                ctx.fillStyle = `rgba(255,255,255,${fm1})`;
                ctx.fillText("Step 1: R̂ₜ = α + β·Rₘₜ  (Expected Return)", 20, chartH + 35);
                
                ctx.fillStyle = `rgba(255,255,255,${fm2})`;
                ctx.fillText("Step 2: CARᵢ = Σ(Rᵢₜ − R̂ᵢₜ)", 20, chartH + 65);
                
                ctx.fillStyle = isPositive ? `rgba(0,255,170,${fm3})` : `rgba(255,100,100,${fm3})`;
                if (isPositive) {
                    ctx.fillText("Step 3: = −0.008 + 0.021 + 0.034 + ... = +0.047 (Value Created)", 20, chartH + 95);
                } else {
                    ctx.fillText("Step 3: = +0.002 − 0.015 − 0.008 + ... = −0.021 (Value Destroyed)", 20, chartH + 95);
                }
            }
        }

        function animate(timestamp) {
            if (!startTime) startTime = timestamp;
            const elapsed = timestamp - startTime;
            const progress = Math.min(1.0, elapsed / DURATION);
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = COLORS.bg;
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            draw(progress, ctx, canvas);
            
            if (progress < 1.0) {
                requestAnimationFrame(animate);
            }
        }

        requestAnimationFrame(animate);
        </script>
        </body>
        </html>
        """
        st.components.v1.html(html_str, height=400)
        
        st.latex(r"\hat{R}_{it} = \alpha_i + \beta_i \cdot R_{mt}")
        st.latex(r"CAR_i = \sum_{t=-3}^{+3} \left(R_{it} - \hat{R}_{it}\right)")
        
        st.info("💡 **Concept:** The CAR strips out market-wide noise to isolate the deal's specific effect. A positive CAR means the market believed value was created — the stock outperformed what CAPM predicted it would do in those six days. This is our training label: 1 if CAR > 0 (value created), 0 otherwise.")
        
        with st.expander("📂 View Source Code (src/features/compute_car.py)"):
            st.code('''def calculate_car(stock_returns, market_returns, event_date):
    """
    Computes Cumulative Abnormal Return (CAR) using the Market Model (CAPM).
    Event window: [-3, +3] trading days around the announcement.
    Estimation window: 252 days ending 30 days prior.
    """
    # 1. Fit OLS in estimation window
    y = stock_returns.loc[estim_start : estim_end]
    X = sm.add_constant(market_returns.loc[estim_start : estim_end])
    model = sm.OLS(y, X).fit()
    
    # 2. Predict expected returns in event window
    X_event = sm.add_constant(market_returns.loc[event_start : event_end])
    R_expected = model.predict(X_event)
    R_actual = stock_returns.loc[event_start : event_end]
    
    # 3. Sum the abnormal returns (Actual - Expected)
    abnormal_returns = R_actual - R_expected
    car_value = abnormal_returns.sum()
    
    return car_value
''', language="python")

    elif stage_idx == 2:
        top_5_json = json.dumps(betweenness_top10[:5])
        html_str = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
          body {{ margin: 0; padding: 0; background-color: #0e1117; overflow: hidden; font-family: monospace; }}
          canvas {{ display: block; width: 100%; height: 380px; cursor: pointer; }}
          #tooltip {{
             position: absolute; display: none; background: #1c1b19; 
             border: 1px solid #00FFAA; color: #cdccca; 
             padding: 8px 12px; border-radius: 6px; font-size: 12px;
             pointer-events: none; white-space: nowrap; box-shadow: 0 0 10px rgba(0,255,170,0.3);
          }}
        </style>
        </head>
        <body>
        <div id="tooltip"></div>
        <canvas id="stage-canvas"></canvas>
        <script>
        const canvas = document.getElementById('stage-canvas');
        const ctx = canvas.getContext('2d');
        const tooltip = document.getElementById('tooltip');

        function resize() {{
            canvas.width = canvas.offsetWidth;
            canvas.height = 380;
        }}
        window.addEventListener('resize', resize);
        resize();

        const COLORS = {{
          bg: '#0e1117',
          accent: '#00FFAA',
          financial: '#5591c7',
          textual: '#fdab43',
          graph: '#00FFAA',
          muted: '#444',
          surface: '#1c1b19'
        }};

        let startTime = null;
        const ANIM_DURATION = 10000;
        const FULL_DURATION = 11000;
        const TOP_NODES = {top_5_json};

        // Layout nodes
        let nodes = [];
        const w = canvas.width;
        
        // Tier 0: Suppliers (y=80)
        nodes.push({{id: 0, x: w*0.25, y: 80, role: '2-hop'}});
        nodes.push({{id: 1, x: w*0.5, y: 80, role: '1-hop'}});
        nodes.push({{id: 2, x: w*0.75, y: 80, role: '2-hop'}});
        
        // Tier 1: Companies (y=190)
        nodes.push({{id: 3, x: w*0.2, y: 190, role: 'other'}});
        nodes.push({{id: 4, x: w*0.4, y: 190, role: 'target'}});
        nodes.push({{id: 5, x: w*0.6, y: 190, role: 'other'}});
        nodes.push({{id: 6, x: w*0.8, y: 190, role: 'other'}});
        
        // Tier 2: Customers (y=300)
        nodes.push({{id: 7, x: w*0.16, y: 300, role: '2-hop'}});
        nodes.push({{id: 8, x: w*0.33, y: 300, role: '1-hop'}});
        nodes.push({{id: 9, x: w*0.5, y: 300, role: 'other'}});
        nodes.push({{id: 10, x: w*0.66, y: 300, role: 'other'}});
        nodes.push({{id: 11, x: w*0.83, y: 300, role: 'other'}});
        
        // Match them to TOP_NODES for tooltip data
        // Target = 0, 1-hops = 1,2, 2-hops = 3,4
        nodes.forEach(n => {{
            if (n.id === 4 && TOP_NODES[0]) n.data = TOP_NODES[0];
            else if (n.id === 1 && TOP_NODES[1]) n.data = TOP_NODES[1];
            else if (n.id === 8 && TOP_NODES[2]) n.data = TOP_NODES[2];
            else if (n.id === 0 && TOP_NODES[3]) n.data = TOP_NODES[3];
            else if (n.id === 2 && TOP_NODES[4]) n.data = TOP_NODES[4];
            else n.data = null;
        }});

        // Edges (from, to)
        const edges = [
            [0, 1], [2, 1], // 2-hops to 1-hop supplier
            [1, 4], [8, 4], // 1-hops to target
            [7, 8],         // 2-hop to 1-hop customer
            [3, 8], [5, 1], [9, 5], [10, 6], [2, 6] // other connections to make it a network
        ];

        // Click interaction
        canvas.addEventListener('click', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            let clickedNode = null;
            for(let n of nodes) {{
                let dx = mouseX - n.x;
                let dy = mouseY - n.y;
                if(dx*dx + dy*dy <= 400) {{ // r=20
                    clickedNode = n;
                    break;
                }}
            }}
            
            if (clickedNode) {{
                let lbl = clickedNode.id === 4 ? "Acquirer Node" : (clickedNode.role !== 'other' ? "Supply Chain Partner" : "Peripheral Node");
                let btw = clickedNode.data ? `${{clickedNode.data.percentile}} %ile` : "< 50th %ile";
                
                tooltip.innerHTML = `<b style="color:#00FFAA">${{lbl}}</b><br/>Betweenness Centrality: ${{btw}}`;
                tooltip.style.display = 'block';
                
                let tw = tooltip.offsetWidth;
                let cRect = canvas.getBoundingClientRect();
                
                let leftPos = e.clientX + 15;
                if (leftPos + tw > cRect.right) {{
                    leftPos = e.clientX - tw - 15;
                }}
                tooltip.style.left = leftPos + 'px';
                tooltip.style.top = (e.clientY - 20) + 'px';
            }} else {{
                tooltip.style.display = 'none';
            }}
        }});
        
        // Hide tooltip on global unclick
        window.addEventListener('click', (e) => {{
            if (e.target !== canvas) tooltip.style.display = 'none';
        }});

        function smoothStep(edge0, edge1, x) {{
            let t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
            return t * t * (3 - 2 * t);
        }}

        function draw(progress, elapsed, ctx) {{
            // Smooth Sub-phases:
            // 0.0-0.2: target pulses
            // 0.2-0.5: 1-hop flowing paths
            // 0.5-0.75: target updates interior vector
            // 0.75-0.95: 2-hop flowing paths
            
            const targetColor = COLORS.graph;
            const hop1Color = COLORS.textual;
            const hop2Color = COLORS.financial;
            const defaultColor = COLORS.muted;
            
            // Draw Edges
            ctx.lineWidth = 1.6;
            edges.forEach(edge => {{
                let n1 = nodes[edge[0]];
                let n2 = nodes[edge[1]];
                
                ctx.beginPath();
                ctx.moveTo(n1.x, n1.y);
                ctx.lineTo(n2.x, n2.y);
                
                let is1Hop = (n2.id === 4 && (n1.id === 1 || n1.id === 8));
                let is2Hop = ((n2.id === 1 && (n1.id === 0 || n1.id === 2)) || (n2.id === 8 && n1.id === 7));
                
                if (is1Hop && progress >= 0.2 && progress < 0.75) {{
                    ctx.strokeStyle = hop1Color;
                    ctx.setLineDash([10, 10]);
                    ctx.lineDashOffset = -(elapsed * 0.04) % 20; // Silky smooth sub-pixel scroll
                    let alpha = smoothStep(0.2, 0.25, progress) * (1 - smoothStep(0.7, 0.75, progress));
                    ctx.globalAlpha = alpha;
                    ctx.stroke();
                    ctx.globalAlpha = 1.0;
                }} else if (is2Hop && progress >= 0.75 && progress < 0.98) {{
                    ctx.strokeStyle = hop2Color;
                    ctx.setLineDash([10, 10]);
                    ctx.lineDashOffset = -(elapsed * 0.04) % 20;
                    let alpha = smoothStep(0.75, 0.8, progress) * (1 - smoothStep(0.95, 0.98, progress)) * 0.8;
                    ctx.globalAlpha = alpha;
                    ctx.stroke();
                    ctx.globalAlpha = 1.0;
                }} else {{
                    ctx.strokeStyle = `rgba(68, 68, 68, 0.3)`;
                    ctx.setLineDash([]);
                    ctx.stroke();
                }}
            }});
            ctx.setLineDash([]);

            // Draw Nodes
            nodes.forEach(n => {{
                let r = 16;
                let c = defaultColor;
                
                // Base background for fading in color on top securely
                ctx.beginPath();
                ctx.arc(n.x, n.y, r, 0, Math.PI*2);
                ctx.fillStyle = defaultColor;
                ctx.fill();
                
                let opac = 0.0;
                if (n.id === 4) {{ 
                    if (progress < 0.98) {{ c = targetColor; opac = 1.0; }} 
                    else {{ c = targetColor; opac = 1 - smoothStep(0.98, 1.0, progress); }}
                }} else if (n.role === '1-hop') {{
                    if (progress >= 0.2 && progress < 0.98) {{
                        c = hop1Color;
                        opac = smoothStep(0.2, 0.25, progress);
                    }}
                }} else if (n.role === '2-hop') {{
                    if (progress >= 0.75 && progress < 0.98) {{
                        c = hop2Color;
                        opac = smoothStep(0.75, 0.8, progress) * 0.8;
                    }}
                }}
                
                if (opac > 0) {{
                    ctx.beginPath();
                    ctx.arc(n.x, n.y, r, 0, Math.PI*2);
                    ctx.fillStyle = c;
                    ctx.globalAlpha = opac;
                    ctx.fill();
                    ctx.globalAlpha = 1.0;
                }}
                
                // Target Ripple (0.0 - 0.2)
                if (n.id === 4 && progress < 0.2) {{
                    let rip = progress / 0.2;
                    ctx.beginPath();
                    ctx.arc(n.x, n.y, r + rip*25, 0, Math.PI*2);
                    ctx.strokeStyle = `rgba(0, 255, 170, ${{Math.max(0, 1 - rip)}})`;
                    ctx.lineWidth = 2.5;
                    ctx.stroke();
                }}
                
                // Target interior graph vector update (0.5 - 0.98)
                if (n.id === 4 && progress >= 0.5 && progress < 0.98) {{
                    ctx.beginPath();
                    ctx.arc(n.x, n.y, r-1, 0, Math.PI*2);
                    ctx.fillStyle = COLORS.surface;
                    ctx.fill();
                    ctx.strokeStyle = targetColor;
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    
                    ctx.fillStyle = targetColor;
                    let barH_base = 6;
                    let animTime = smoothStep(0.5, 0.55, progress);
                    for(let b=0; b<5; b++) {{
                        let h = barH_base + Math.sin(b*1.5 + elapsed*0.003)*4.5 * animTime;
                        ctx.fillRect(n.x - 9 + b*4, n.y + 6 - Math.max(0, h), 2.5, Math.max(0, h));
                    }}
                }}
            }});
            
            // Master Label for animation stage
            ctx.fillStyle = "#FFF";
            ctx.font = "16px monospace";
            ctx.textAlign = "center";
            let labelText = "";
            let stageTime = progress * 10.0;
            
            if (stageTime >= 0 && stageTime < 2.0) labelText = "Initialising Graph Convolution";
            else if (stageTime >= 2.0 && stageTime < 5.0) labelText = "Aggregating 1-hop neighbours...";
            else if (stageTime >= 5.0 && stageTime < 7.5) labelText = "h⁽¹⁾ updated with aggregate signal";
            else if (stageTime >= 7.5 && stageTime < 9.5) labelText = "Layer 2: 2-hop aggregation";
            else labelText = "Graph topology compressed to vector";
            
            ctx.fillText(labelText, w/2, 40);
        }}

        function animate(timestamp) {{
            if (!startTime) startTime = timestamp;
            const elapsed = timestamp - startTime;
            const progress = Math.min(1.0, (elapsed % FULL_DURATION) / ANIM_DURATION);
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = COLORS.bg;
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            draw(progress, elapsed, ctx);
            
            requestAnimationFrame(animate);
        }}

        requestAnimationFrame(animate);
        </script>
        </body>
        </html>
        """
        st.components.v1.html(html_str, height=400)
        
        st.latex(r"h_v^{(k)} = \sigma\!\left(W \cdot \mathrm{MEAN}\!\left\{h_u^{(k-1)} : u \in \mathcal{N}(v)\right\}\right)")
        
        st.info("💡 **Concept:** GraphSAGE asks: 'What do my neighbours know?' Each company's embedding is enriched by aggregating its supply chain partners' features. A firm deeply embedded in a critical supply chain carries structural information that no balance sheet can capture — and this is exactly what the graph predicts that financial ratios alone cannot.")
        
        with st.expander("📂 View Source Code (src/features/build_hetero_graph.py)"):
            st.code('''import dgl
import dgl.function as fn
import torch
import torch.nn as nn

class GraphSAGELayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GraphSAGELayer, self).__init__()
        self.linear = nn.Linear(in_feats * 2, out_feats)

    def forward(self, g, feature):
        # 1. Message passing: aggregate neighbors (MEAN)
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
            neigh_feature = g.ndata['neigh']
            
            # 2. Update step: concatenate self and neighbor features, apply linear + activation
            h_concat = torch.cat([feature, neigh_feature], dim=1)
            h_new = torch.relu(self.linear(h_concat))
            return h_new
''', language="python")

        with st.expander("📊 Stage Output Statistics"):
            st.markdown("- **5,730** nodes | **7,582** supply edges")
            st.markdown("- **GraphSAGE standalone AUC:** 0.8245")
            st.markdown("- **Delta vs M1 baseline:** +0.024")

    elif stage_idx == 3:
        html_str = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
          body { margin: 0; padding: 0; background-color: #0e1117; overflow: hidden; font-family: monospace; }
          canvas { display: block; width: 100%; height: 380px; }
        </style>
        </head>
        <body>
        <canvas id="stage-canvas"></canvas>
        <script>
        const canvas = document.getElementById('stage-canvas');
        const ctx = canvas.getContext('2d');

        function resize() {
            canvas.width = canvas.offsetWidth;
            canvas.height = 380;
        }
        window.addEventListener('resize', resize);
        resize();

        const COLORS = {
          bg: '#0e1117',
          accent: '#00FFAA',
          surface: '#1c1b19',
          muted: '#444',
          text: '#cdccca'
        };

        let startTime = null;
        const ANIM_DURATION = 10000;
        const FULL_DURATION = 11000; // 1s wait
        
        const words = ["The", "acquisition", "will", "create", "significant", "synergy", "value", "for", "shareholders", "[CLS]"];
        
        const attn = [];
        for(let r=0; r<10; r++) {
            attn[r] = [];
            for(let c=0; c<10; c++) {
                let randomVal = ((r * 13 + c * 7 + 3) % 11) / 10.0;
                let val = randomVal * 0.3; 
                if (c===1 || c===5 || c===6) val += 0.5 + randomVal*0.2;
                if (r===c) val += 0.4;
                attn[r][c] = Math.min(1.0, val);
            }
        }

        function getHeatColor(val) {
            let r,g,b;
            if (val < 0.5) {
                let v = val * 2;
                r = 26 + v*(85-26); g = 26 + v*(145-26); b = 46 + v*(199-46);
            } else {
                let v = (val - 0.5) * 2;
                r = 85 + v*(0-85); g = 145 + v*(255-145); b = 199 + v*(170-199);
            }
            return `rgb(${Math.floor(r)},${Math.floor(g)},${Math.floor(b)})`;
        }
        
        function drawArc(i1, i2, PADDING, canvasW, alpha) {
            let cx1 = PADDING + i1 * 50 + 25;
            let cx2 = PADDING + i2 * 50 + 25;
            let cy = 40;
            
            ctx.beginPath();
            ctx.moveTo(cx1, cy);
            ctx.quadraticCurveTo((cx1+cx2)/2, cy - 30, cx2, cy);
            
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = `rgba(0, 255, 170, ${alpha})`;
            ctx.stroke();
        }

        function smoothStep(edge0, edge1, x) {
            let t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
            return t * t * (3 - 2 * t);
        }

        function draw(progress, elapsed, ctx, canvas) {
            const w = canvas.width;
            const h = canvas.height;
            const PADDING = (w - 10 * 50) / 2;
            
            // Phase 1: Tokens (0.0 to 0.15)
            let tokenProg = Math.min(1.0, progress / 0.15);
            let tokensToShow = Math.floor(tokenProg * 10);
            
            ctx.font = "12px monospace";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            
            for(let i=0; i<10; i++) {
                if (i <= tokensToShow) {
                    let cx = PADDING + i * 50 + 25;
                    let cy = 60;
                    
                    ctx.fillStyle = COLORS.surface;
                    ctx.strokeStyle = COLORS.muted;
                    ctx.beginPath();
                    if (ctx.roundRect) ctx.roundRect(cx - 22, cy - 12, 44, 24, 4);
                    else ctx.rect(cx - 22, cy - 12, 44, 24);
                    ctx.fill();
                    ctx.stroke();
                    
                    ctx.fillStyle = COLORS.text;
                    ctx.fillText(words[i], cx, cy);
                }
            }
            
            // Phase 2: Heatmap (0.15 to 0.40)
            let heatProg = Math.min(1.0, Math.max(0, (progress - 0.15) / 0.25));
            let cellsToShow = Math.floor(heatProg * 100);
            let sqSize = 14;
            let heatY = 100;
            let heatX = w/2 - (10 * sqSize)/2;
            
            if (heatProg > 0) {
                for(let r=0; r<10; r++) {
                    for(let c=0; c<10; c++) {
                        let idx = r * 10 + c;
                        if (idx < cellsToShow) {
                            ctx.fillStyle = getHeatColor(attn[r][c]);
                            ctx.fillRect(heatX + c*sqSize, heatY + r*sqSize, sqSize-1, sqSize-1);
                        }
                    }
                }
                
                if (heatProg === 1.0) {
                    ctx.fillStyle = COLORS.muted;
                    ctx.fillText("Q", heatX - 20, heatY + 5*sqSize);
                    ctx.fillText("Kᵀ", heatX + 5*sqSize, heatY - 10);
                }
            }
            
            // Phase 3: Arcs (0.40 to 0.60)
            let arcProg = Math.max(0, Math.min(1.0, (progress - 0.40) / 0.20));
            if (arcProg > 0) {
                drawArc(1, 5, PADDING, w, 0.7 * arcProg);
                drawArc(6, 8, PADDING, w, 0.7 * arcProg);
            }
            
            // Phase 4: PCA (0.60 to 0.90)
            let pcaProg = smoothStep(0.60, 0.90, progress);
            if (progress >= 0.60) {
                let baseY = 290;
                let barH = 40;
                
                let numBarsStart = 768;
                let numBarsEnd = 32;
                let spanStart = w * 0.9;
                let spanEnd = w * 0.25;
                
                let currentNum = numBarsStart - Math.floor(pcaProg * (numBarsStart - numBarsEnd));
                let currentSpan = spanStart - (pcaProg * (spanStart - spanEnd));
                let currentW = 1.0 + (pcaProg * 3.0);
                
                let startX = (w - currentSpan) / 2;
                let step = currentSpan / currentNum;
                
                ctx.fillStyle = `rgba(0, 255, 170, ${0.4 + 0.6*pcaProg})`;
                
                for(let i=0; i<currentNum; i++) {
                    ctx.fillRect(startX + i*step, baseY, currentW, barH);
                }
                
                ctx.fillStyle = COLORS.text;
                ctx.font = "14px monospace";
                ctx.fillText(`Compressing ${currentNum} dims → 32 PCA dims`, w/2, baseY + barH + 20);
            }
        }

        function animate(timestamp) {
            if (!startTime) startTime = timestamp;
            const elapsed = timestamp - startTime;
            const progress = Math.min(1.0, (elapsed % FULL_DURATION) / ANIM_DURATION);
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = COLORS.bg;
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            draw(progress, elapsed, ctx, canvas);
            
            requestAnimationFrame(animate);
        }

        requestAnimationFrame(animate);
        </script>
        </body>
        </html>
        """
        st.components.v1.html(html_str, height=400)
        
        st.latex(r"\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)\!V")
        
        st.info("💡 **Concept:** FinBERT was pre-trained on 4.9B words of financial text. Unlike generic BERT, it knows that 'synergy' in an M&A filing has a specific meaning. The attention mechanism learns which words in the MD&A are predictive of deal outcomes — then PCA compresses 768 embedding dimensions down to 32 uncorrelated components that efficiently encode management sentiment.")
        
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("🧩 Why strictly use the [CLS] Token?"):
                st.write("In BERT architectures, the `[CLS]` (Classification) token is prepended to every text sequence. Governed by multi-head self-attention during pre-training, it learns to aggregate the deep contextual syntax and global sentiment of the entire document into a single unified 768-dimensional vector, operating as the definitive representation for downstream prediction.")
        with col2:
            with st.expander("🔬 Why reduce dimensions with PCA?"):
                st.write("Raw 768-D embeddings contain significant collinearity (redundant stylistic variance). By retaining only the exact top 32 Principal Components, we mathematically distil the pure orthogonal signals of management intent while structurally protecting the final XGBoost estimator from the curse of dimensionality and catastrophic overfitting.")
        
        with st.expander("📂 View Source Code (src/features/run_text_features.py)"):
            st.code('''from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
import torch

def extract_finbert_embeddings(text_series, max_len=512):
    """
    Passes MD&A docs through FinBERT to generate 768-D representation,
    then compresses dimensionality using PCA.
    """
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    model = BertModel.from_pretrained('yiyanghkust/finbert-tone')
    
    embeddings = []
    # 1. Retrieve 768-dimensional token attention outputs
    for text in text_series:
        inputs = tokenizer(text, return_tensors='pt', padding='max_length', 
                           truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use [CLS] token embedding
        cls_emb = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls_emb)
        
    X_text = np.vstack(embeddings)
    
    # 2. Extract strictly 32 principal components to capture orthogonal variance
    pca = PCA(n_components=32, random_state=42)
    X_text_pca = pca.fit_transform(X_text)
    
    return X_text_pca
''', language="python")

        with st.expander("📊 Stage Output Statistics"):
            st.markdown("- **1,674** text-coverage deals")
            st.markdown("- **768 → 32** PCA dimensionality reduction")

    elif stage_idx == 4:
        html_str = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
          body { margin: 0; padding: 0; background-color: #0e1117; overflow: hidden; font-family: monospace; }
          canvas { display: block; width: 100%; height: 380px; }
        </style>
        </head>
        <body>
        <canvas id="stage-canvas"></canvas>
        <script>
        const canvas = document.getElementById('stage-canvas');
        const ctx = canvas.getContext('2d');

        function resize() {
            canvas.width = canvas.offsetWidth;
            canvas.height = 380;
        }
        window.addEventListener('resize', resize);
        resize();

        const COLORS = {
          bg: '#0e1117',
          graph: '#00FFAA',
          textual: '#fdab43',
          financial: '#5591c7',
          surface: '#1c1b19',
          muted: '#444',
          text: '#cdccca'
        };

        let startTime = null;
        let particles = [];
        let lastSpawn = 0;

        function getBezier(t, x0, y0, cx1, cy1, cx2, cy2, x1, y1) {
            let u = 1 - t;
            let tt = t * t;
            let uu = u * u;
            let uuu = uu * u;
            let ttt = tt * t;

            let x = uuu * x0 + 3 * uu * t * cx1 + 3 * u * tt * cx2 + ttt * x1;
            let y = uuu * y0 + 3 * uu * t * cy1 + 3 * u * tt * cy2 + ttt * y1;
            return {x, y};
        }

        function draw(elapsed, ctx, canvas) {
            const w = canvas.width;
            
            const leftX = w * 0.18;
            const rightX = w * 0.82;
            const boxW = 160;
            const boxH = 46;
            
            const yGraph = 80;
            const yText = 190;
            const yFin = 300;
            const yFusion = 190;
            
            // Spawn particles
            if (elapsed - lastSpawn > 80) {
                lastSpawn = elapsed;
                particles.push({ t: 0, c: COLORS.graph, yStart: yGraph });
                particles.push({ t: 0, c: COLORS.textual, yStart: yText });
                particles.push({ t: 0, c: COLORS.financial, yStart: yFin });
            }
            
            for (let i = particles.length - 1; i >= 0; i--) {
                let p = particles[i];
                p.t += 0.012; // particle speed over duration
                if (p.t > 1) {
                    particles.splice(i, 1);
                }
            }
            
            // Draw flowing background tubes/wires
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 10]);
            ctx.lineDashOffset = -(elapsed * 0.02) % 15;
            
            [yGraph, yText, yFin].forEach(yS => {
                ctx.beginPath();
                ctx.moveTo(leftX + boxW/2, yS);
                ctx.bezierCurveTo(leftX + boxW/2 + 100, yS, rightX - boxW/2 - 100, yFusion, rightX - boxW/2, yFusion);
                ctx.strokeStyle = `rgba(68, 68, 68, 0.4)`;
                ctx.stroke();
            });
            ctx.setLineDash([]);
            
            // Draw particles
            particles.forEach(p => {
                let pos = getBezier(p.t, leftX + boxW/2, p.yStart, leftX + boxW/2 + 100, p.yStart, rightX - boxW/2 - 100, yFusion, rightX - boxW/2, yFusion);
                
                // Particle streak
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, 4, 0, Math.PI*2);
                ctx.fillStyle = p.c;
                // Fade in near spawn, fade out near target bounds
                let opac = Math.sin(p.t * Math.PI);
                ctx.globalAlpha = opac;
                ctx.fill();
                
                // Outer glow
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, 10, 0, Math.PI*2);
                ctx.fillStyle = p.c;
                ctx.globalAlpha = opac * 0.4;
                ctx.fill();
            });
            ctx.globalAlpha = 1.0;
            
            // Target Fusion Box
            let pulseScale = 1.0 + 0.06 * Math.sin(elapsed * 0.005);
            ctx.fillStyle = COLORS.surface;
            ctx.strokeStyle = '#FFF';
            ctx.lineWidth = 2;
            
            let tfW = boxW * pulseScale;
            let tfH = boxH * 1.5 * pulseScale;
            ctx.beginPath();
            if(ctx.roundRect) ctx.roundRect(rightX - tfW/2, yFusion - tfH/2, tfW, tfH, 6);
            else ctx.rect(rightX - tfW/2, yFusion - tfH/2, tfW, tfH);
            ctx.fill();
            ctx.stroke();
            
            ctx.fillStyle = '#FFF';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.font = `bold ${14 * pulseScale}px monospace`;
            ctx.fillText("Master Dataset", rightX, yFusion - 8);
            ctx.fillStyle = COLORS.accent;
            ctx.font = `${12 * pulseScale}px monospace`;
            ctx.fillText("(249d)", rightX, yFusion + 10);
            
            // Source Boxes (Left)
            function drawBox(x, y, label, labelC, dims) {
                ctx.fillStyle = COLORS.surface;
                ctx.strokeStyle = labelC;
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                if(ctx.roundRect) ctx.roundRect(x - boxW/2, y - boxH/2, boxW, boxH, 4);
                else ctx.rect(x - boxW/2, y - boxH/2, boxW, boxH);
                ctx.fill();
                ctx.stroke();
                
                ctx.fillStyle = COLORS.text;
                ctx.font = "12px monospace";
                ctx.fillText(label, x, y - 6);
                ctx.fillStyle = labelC;
                ctx.fillText(`(${dims})`, x, y + 8);
            }
            
            drawBox(leftX, yGraph, "GraphSAGE Topology", COLORS.graph, "64d");
            drawBox(leftX, yText, "FinBERT PCA", COLORS.textual, "32d");
            drawBox(leftX, yFin, "Financial Ratios", COLORS.financial, "153d");
        }

        function animate(timestamp) {
            if (!startTime) startTime = timestamp;
            const elapsed = timestamp - startTime;
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = COLORS.bg;
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            draw(elapsed, ctx, canvas);
            
            requestAnimationFrame(animate);
        }

        requestAnimationFrame(animate);
        </script>
        </body>
        </html>
        """
        st.components.v1.html(html_str, height=400)
        
        st.latex(r"X_{\mathrm{fused}} = \left[ h_v^{\mathrm{graph}} \parallel h_v^{\mathrm{text}} \parallel x_v^{\mathrm{fin}} \right]")
        
        st.info("💡 **Concept:** This is where the magic happens. We concatenate the structured fundamental financials with the unstructured supply-chain network topology and MD&A management sentiment. This creates a dense **249-dimensional** multimodal master feature matrix, ensuring the final estimator evaluates the holistic picture: the math, the network, and the narrative.")
        
        with st.expander("📂 View Source Code (src/data/make_dataset.py)"):
            st.code('''import pandas as pd

def build_multimodal_dataset(financial_df, graph_embeddings, text_embeddings):
    """
    Concatenates strictly aligned feature structures into a single fused space.
    """
    # 1. Merge graph topology vectors (64d)
    df = pd.merge(financial_df, graph_embeddings, on='deal_id', how='left')
    
    # 2. Merge language vectors (32d)
    df = pd.merge(df, text_embeddings, on='deal_id', how='left')
    
    # Drop primary index keys to yield continuous feature spaces
    X_fused = df.drop(columns=['deal_id', 'target', 'date'])
    y = df['target']
    
    return X_fused, y
''', language="python")

    elif stage_idx == 5:
        html_str = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
          body { margin: 0; padding: 0; background-color: #0e1117; overflow: hidden; font-family: monospace; }
          canvas { display: block; width: 100%; height: 380px; }
        </style>
        </head>
        <body>
        <canvas id="stage-canvas"></canvas>
        <script>
        const canvas = document.getElementById('stage-canvas');
        const ctx = canvas.getContext('2d');

        function resize() {
            canvas.width = canvas.offsetWidth;
            canvas.height = 380;
        }
        window.addEventListener('resize', resize);
        resize();

        const COLORS = {
          bg: '#0e1117',
          accent: '#00FFAA',
          surface: '#1c1b19',
          muted: '#444',
          text: '#cdccca'
        };

        let startTime = null;
        const ANIM_DURATION = 8000;
        const FULL_DURATION = 9000;

        function smoothStep(edge0, edge1, x) {
            let t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
            return t * t * (3 - 2 * t);
        }

        // define tree
        let nodes = [
            { id: 0, text: "SYNERGY_TFIDF > 0.12", x: 0.5, y: 70, w: 180, parent: null }, 
            
            { id: 1, text: "EBITDA Margin < 0.05", x: 0.25, y: 160, w: 160, parent: 0, pathText: "No" }, 
            { id: 2, text: "Betweenness < 0.8",    x: 0.75, y: 160, w: 160, parent: 0, pathText: "Yes" }, 
            
            { id: 3, text: "Failure (-3.2% CAR)", x: 0.12, y: 260, w: 140, parent: 1, pathText: "Yes", isLeaf: true, color: '#ff6464' },
            { id: 4, text: "Neutral (+0.5% CAR)", x: 0.38, y: 260, w: 140, parent: 1, pathText: "No", isLeaf: true, color: '#444' },
            { id: 5, text: "Neutral (+1.2% CAR)", x: 0.62, y: 260, w: 140, parent: 2, pathText: "Yes", isLeaf: true, color: '#444' },
            { id: 6, text: "Success (+4.1% CAR)", x: 0.88, y: 260, w: 140, parent: 2, pathText: "No", isLeaf: true, color: '#00FFAA' }
        ];

        let edges = [
            { from: 0, to: 1, drawStart: 0.1, drawEnd: 0.25 },
            { from: 0, to: 2, drawStart: 0.1, drawEnd: 0.25 },
            
            { from: 1, to: 3, drawStart: 0.4, drawEnd: 0.55 },
            { from: 1, to: 4, drawStart: 0.4, drawEnd: 0.55 },
            { from: 2, to: 5, drawStart: 0.4, drawEnd: 0.55 },
            { from: 2, to: 6, drawStart: 0.4, drawEnd: 0.55 }
        ];

        // node visibility timing
        nodes[0].visStart = 0.0; nodes[0].visEnd = 0.1;
        nodes[1].visStart = 0.25; nodes[1].visEnd = 0.4;
        nodes[2].visStart = 0.25; nodes[2].visEnd = 0.4;
        nodes[3].visStart = 0.55; nodes[3].visEnd = 0.7;
        nodes[4].visStart = 0.55; nodes[4].visEnd = 0.7;
        nodes[5].visStart = 0.55; nodes[5].visEnd = 0.7;
        nodes[6].visStart = 0.55; nodes[6].visEnd = 0.7;

        function draw(progress, elapsed, ctx, canvas) {
            const w = canvas.width;
            
            ctx.lineWidth = 2;
            ctx.font = "12px monospace";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            
            // Draw edges
            edges.forEach(e => {
                let n1 = nodes[e.from];
                let n2 = nodes[e.to];
                
                let p = smoothStep(e.drawStart, e.drawEnd, progress); // 0 to 1
                if (p > 0) {
                    let startX = n1.x * w;
                    let startY = n1.y + 15;
                    let endX = n2.x * w;
                    let endY = n2.y - 15;
                    
                    let currX = startX + (endX - startX) * p;
                    let currY = startY + (endY - startY) * p;
                    
                    ctx.beginPath();
                    ctx.moveTo(startX, startY);
                    ctx.lineTo(currX, currY);
                    ctx.strokeStyle = 'rgba(68, 68, 68, 0.5)';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    
                    // Highlight Flow Pulse for successful node path
                    let isHighlightPath = progress >= 0.7 && ((e.from===0 && e.to===2) || (e.from===2 && e.to===6));
                    
                    if (isHighlightPath) {
                        ctx.beginPath();
                        ctx.moveTo(startX, startY);
                        ctx.lineTo(currX, currY);
                        ctx.strokeStyle = 'rgba(0, 255, 170, 0.2)';
                        ctx.lineWidth = 4;
                        ctx.stroke();
                        
                        // Dashed tracing laser
                        ctx.setLineDash([10, 10]);
                        ctx.lineDashOffset = -(elapsed * 0.05) % 20;
                        ctx.strokeStyle = COLORS.accent;
                        ctx.lineWidth = 2;
                        ctx.stroke();
                        ctx.setLineDash([]);
                    }
                    
                    // Draw YES/NO bounding text labels
                    if (progress > e.drawEnd) {
                        ctx.fillStyle = COLORS.text;
                        ctx.font = "10px monospace";
                        
                        let mx = startX + (endX - startX) * 0.5;
                        let my = startY + (endY - startY) * 0.5 - 10;
                        
                        if (isHighlightPath) {
                            ctx.fillStyle = COLORS.accent;
                            ctx.font = "bold 10px monospace";
                        }
                        
                        ctx.fillText(n2.pathText, mx, my);
                    }
                }
            });
            
            // Draw nodes
            ctx.font = "12px monospace";
            nodes.forEach(n => {
                let p = smoothStep(n.visStart, n.visEnd, progress);
                if (p > 0) {
                    let boxW = n.w;
                    let boxH = 30;
                    let nx = n.x * w;
                    
                    let isHighlightTarget = false;
                    if (progress >= 0.7 && (n.id === 0 || n.id === 2 || n.id === 6)) {
                        isHighlightTarget = true;
                    }
                    
                    let bgC = COLORS.surface;
                    let strokeC = COLORS.muted;
                    let textC = COLORS.text;
                    let tFont = "12px monospace";
                    
                    // State mutations
                    if (n.isLeaf && progress >= 0.8) {
                        if (n.id === 3) {
                            bgC = "rgba(255, 100, 100, 0.15)";
                            strokeC = '#ff6464';
                            textC = '#ff6464';
                            tFont = "bold 12px monospace";
                        } else if (n.id === 6) {
                            bgC = "rgba(0, 255, 170, 0.15)";
                            strokeC = COLORS.accent;
                            textC = COLORS.accent;
                            tFont = "bold 12px monospace";
                        }
                    } else if (isHighlightTarget) {
                        bgC = "rgba(0, 255, 170, 0.1)";
                        strokeC = COLORS.accent;
                        if (!n.isLeaf) {
                            textC = '#FFF'; 
                            tFont = "bold 12px monospace";
                        }
                    }
                    
                    ctx.globalAlpha = p; // CSS fade in box
                    
                    ctx.fillStyle = bgC;
                    ctx.strokeStyle = strokeC;
                    ctx.beginPath();
                    if(ctx.roundRect) ctx.roundRect(nx - boxW/2, n.y - boxH/2, boxW, boxH, 4);
                    else ctx.rect(nx - boxW/2, n.y - boxH/2, boxW, boxH);
                    ctx.fill();
                    ctx.stroke();
                    
                    ctx.fillStyle = textC;
                    ctx.font = tFont;
                    ctx.fillText(n.text, nx, n.y);
                    
                    ctx.globalAlpha = 1.0;
                }
            });
            
            // Text annotation overlay on top right indicating processing mechanism
            ctx.textAlign = 'right';
            ctx.fillStyle = COLORS.muted;
            ctx.font = "14px monospace";
            if (progress >= 0.1) {
                let numTree = Math.floor(progress * 500);
                if (numTree > 423) numTree = 423; // Just lock arbitrarily for aesthetics
                ctx.fillText(`Gradient Boosting Step: [${progress >= 0.95 ? 'Complete' : numTree + '/500'}]`, w - 20, 30);
            }
        }

        function animate(timestamp) {
            if (!startTime) startTime = timestamp;
            const elapsed = timestamp - startTime;
            const progress = Math.min(1.0, (elapsed % FULL_DURATION) / ANIM_DURATION);
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = COLORS.bg;
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            draw(progress, elapsed, ctx, canvas);
            
            requestAnimationFrame(animate);
        }

        requestAnimationFrame(animate);
        </script>
        </body>
        </html>
        """
        st.components.v1.html(html_str, height=400)
        
        st.latex(r"\hat{y}_i = \sum_{k=1}^K f_k(X_{\mathrm{fused}, i})")
        
        st.info("💡 **Concept:** The multimodal 249-dimensional dataset is fed into an ensemble of gradient boosted decision trees. Because XGBoost evaluates features non-linearly, it can learn incredibly complex sub-interactions — e.g., *'A strong supply chain topology only drives target synergy if the acquirer has a low EBITDA margin constraint and positive management sentiment'* — mitigating the linear failure mechanisms seen in conventional standard econometric CAPM regressions.")
        
        with st.expander("📂 View Source Code (src/models/train_xgboost.py)"):
            st.code('''import xgboost as xgb
from sklearn.metrics import roc_auc_score

def train_ensemble(X_train, y_train, X_val, y_val):
    """
    Fits the heavily regularised fusion ensemble model to predict 
    positive synergy outcomes based on the multi-dimensional feature space.
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Strictly regularised hyperparameters preventing collinearity overfits
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'eta': 0.05,            # learning rate
        'subsample': 0.8,       # bagging
        'colsample_bytree': 0.8 # feature sampling
    }
    
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    # Execute non-linear ensemble gradient boosting
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=500, 
        evals=evals, 
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # Evaluate generalization stability
    preds = model.predict(dval)
    val_auc = roc_auc_score(y_val, preds)
    print(f"Final Validation AUC: {val_auc:.4f}")
    
    return model
''', language="python")

    else:
        pass
