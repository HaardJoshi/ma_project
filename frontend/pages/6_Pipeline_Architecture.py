import json
import streamlit as st
from frontend.utils import load_betweenness_data, setup_page

setup_page("Pipeline Architecture")

def build_pipeline_html(betweenness_top10):
    html_template = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
:root {{
    --accent: #00FFAA;
    --financial: #5591c7;
    --textual: #fdab43;
    --graph: #00FFAA;
    --bg: #0E1117;
    --surface: #1A1C24;
    --text: #cdccca;
    --muted: #555555;
    --border: rgba(255, 255, 255, 0.08);
}}
body {{
    margin: 0; padding: 10px;
    background: var(--bg);
    color: var(--text);
    font-family: "Courier New", Courier, monospace;
    overflow-y: hidden; /* handle height resizing correctly */
}}
#pipeline-root {{
    display: flex; gap: 20px; align-items: flex-start;
    opacity: 1; transition: opacity 0.5s ease;
}}
.stepper {{
    width: 25%; display: flex; flex-direction: column; position: relative;
    user-select: none;
}}
.step {{
    position: relative; padding: 15px 0 15px 40px; cursor: pointer;
    opacity: 0.5; transition: all 0.3s ease;
    font-size: 14px;
}}
.step:hover {{ opacity: 0.9; }}
.step.active {{
    opacity: 1; color: var(--accent); font-weight: bold;
}}
.step.completed {{
    opacity: 0.7;
}}
.step::before {{
    content: ''; position: absolute; left: 10px; top: 18px;
    width: 12px; height: 12px; border-radius: 50%;
    background: var(--muted); z-index: 2; transition: all 0.3s ease;
}}
.step.active::before {{
    background: var(--bg);
    border: 3px solid var(--accent);
    box-shadow: 0 0 10px var(--accent);
    top: 15px; left: 7px; width: 14px; height: 14px;
}}
.step.completed::before {{
    background: var(--accent); border: 2px solid var(--accent);
    top: 16px; left: 8px; width: 14px; height: 14px;
}}
#stepper-line {{
    position: absolute; left: 16px; top: 25px;
    width: 2px;
    background: repeating-linear-gradient(to bottom, var(--muted) 0, var(--muted) 5px, transparent 5px, transparent 10px);
    z-index: 1;
}}
#stepper-fill {{
    position: absolute; left: 16px; top: 25px;
    width: 2px; background: var(--accent);
    z-index: 1; transition: height 0.4s ease;
}}
.panel {{
    width: 75%; display: flex; flex-direction: column; gap: 15px;
}}
.canvas-wrapper {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; width: 100%; height: 340px;
    position: relative; overflow: hidden;
}}
#main-canvas {{
    width: 100%; height: 100%; display: block;
}}
.box {{
    padding: 20px; background: var(--surface); border-radius: 6px;
    border: 1px solid var(--border);
}}
.formula-box {{
    min-height: 45px; text-align: center; font-size: 1.5em;
    display: flex; align-items: center; justify-content: center;
    color: #FFF; overflow-x: auto;
}}
.concept-box {{
    font-size: 0.9em; min-height: 50px; line-height: 1.6;
    color: #cdccca;
}}
.concept-title {{
    font-weight: bold; color: #FFF; margin-bottom: 8px; font-size: 1.1em;
</style>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
</head>
<body>

<div id="pipeline-root">
    <div class="stepper" id="stepper-list">
        <div id="stepper-line"></div>
        <div id="stepper-fill"></div>
        <!-- Steps updated via JS -->
    </div>
    <div class="panel">
        <div class="canvas-wrapper" style="position:relative;">
            <canvas id="main-canvas" style="position:absolute; top:0; left:0; z-index:1;"></canvas>
            <div id="d3-container" style="position:absolute; top:0; left:0; width:100%; height:100%; z-index:2; display:none;"></div>
            <div id="d3-tooltip" style="position:absolute; background:var(--surface); border:1px solid #00FFAA; padding:5px 10px; border-radius:4px; font-size:12px; display:none; pointer-events:none; z-index:3; color:#cdccca; white-space:nowrap; box-shadow: 0 0 10px rgba(0,255,170,0.3);"></div>
        </div>
        <div class="box formula-box" id="formula-container"></div>
        <div class="box concept-box" id="concept-container"></div>
    </div>
</div>

<script>
// --- Data Injection ---

const BETWEENNESS_DATA = {betweenness_json};

// --- Stage Definitions ---
const STAGES = [
    {{ title: "0. Data Ingestion & Alignment", formula: "Coverage: ~2,864 Core Deals", desc: "Heterogeneous multi-source alignment of Bloomberg M&A feeds, SEC EDGAR text, and SPLC dependency linkages." }},
    {{ title: "1. Target Function Definition", formula: "CAR = \\\\sum_{{t=-5}}^{{+5}} (R_{{it}} - \\\\hat{{R}}_{{it}})", desc: "Cumulative Abnormal Returns over an 11-day event window around the deal announcement. Modeled using OLS against the market benchmark." }},
    {{ title: "2. GraphSAGE Topological Embeds", formula: "h_v^{{(k)}} = \\\\sigma \\\\Big( W \\\\cdot \\\\text{{MEAN}}\\\\{{h_u^{{(k-1)}} : u \\\\in \\\\mathcal{{N}}(v)\\\\}} \\\\Big)", desc: "Information propogation across supply chains. Acquiring entities pass graph messages to learn risk exposure profiles from their supplier networks." }},
    {{ title: "3. FinBERT Semantic Attention", formula: "\\\\text{{Attn}}(Q,K,V) = \\\\text{{softmax}} \\\\left( \\\\frac{{Q K^T}}{{\\\\sqrt{{d_k}}}} \\\\right) V", desc: "Distilled multi-head attention computing congruence vectors for MD&A passages and Risk Factor disclosures of merging firms." }},
    {{ title: "4. Multimodal Early Fusion", formula: "X = [ x_{{fin}} \\\\parallel x_{{text}} \\\\parallel x_{{graph}} ]", desc: "Concatenation of raw financial factors (N=249), text embeddings (N=512), and graph properties into a monolithic heterogeneous vector space." }},
    {{ title: "5. Gradient Boosting Ensemble", formula: "\\\\hat{{y}} = \\\\sum_{{k=1}}^K f_k(X)", desc: "Stratified 5-Fold Cross Validation. Output is a probability calibrated to predict value-accreting rank across all M&A targets in the pool." }}
];

function triggerHeightResize() {{
    const rootBlock = document.getElementById('pipeline-root');
    const height = rootBlock ? rootBlock.scrollHeight + 50 : document.body.scrollHeight + 50;
    window.parent.postMessage({{type: 'streamlit:setFrameHeight', height: height}}, '*');
}}

// --- Canvas Animations ---
const STAGE_ANIMATIONS = {{
    0: {{ start: (ctx, w, h) => ctx.clearRect(0,0,w,h), stop: () => {{}} }},
    // Stage 1: CAR Area Chart
    1: {{
        rafId: null,
        progress: 0,
        start(ctx, w, h) {{
            this.progress = 0;
            this.draw(ctx, w, h);
        }},
        stop() {{
            if (this.rafId) cancelAnimationFrame(this.rafId);
        }},
        draw(ctx, w, h) {{
            this.progress += 0.015;
            if (this.progress > 1.2) this.progress = 1.2;
            ctx.clearRect(0,0,w,h);
            
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, h/2); ctx.lineTo(w, h/2);
            for(let i=1; i<10; i++) {{
                ctx.moveTo(w*(i/10), 0); ctx.lineTo(w*(i/10), h);
            }}
            ctx.stroke();

            const pts = [];
            const numPoints = 80;
            for(let i=0; i<numPoints; i++) {{
                let t = i / (numPoints-1);
                let x = t * w;
                let y = h/2;
                if (t > 0.45 && t < 0.55) {{
                    y -= Math.sin((t-0.45)*Math.PI/0.10) * h*0.35;
                }} else if (t >= 0.55 && t < 0.70) {{
                    y -= Math.sin(1*Math.PI/0.10) * h*0.35;
                    y += (t-0.55) * h*0.5;
                }} else if (t >= 0.70) {{
                    y -= h*0.35;
                    y += (0.15) * h*0.5;
                    y -= (t-0.70) * h*0.2;
                }}
                pts.push({{x, y}});
            }}

            const drawLen = Math.floor(this.progress * numPoints);
            const drawPts = pts.slice(0, Math.max(0, drawLen));
            
            if (drawPts.length > 0) {{
                // Area fill
                ctx.beginPath();
                ctx.moveTo(drawPts[0].x, h/2);
                for(let p of drawPts) ctx.lineTo(p.x, p.y);
                ctx.lineTo(drawPts[drawPts.length-1].x, h/2);
                ctx.closePath();
                
                let gradient = ctx.createLinearGradient(0, 0, 0, h/2);
                gradient.addColorStop(0, 'rgba(0, 255, 170, 0.4)');
                gradient.addColorStop(1, 'rgba(0, 255, 170, 0)');
                ctx.fillStyle = gradient;
                ctx.fill();

                // Line stroke
                ctx.beginPath();
                ctx.strokeStyle = '#00FFAA';
                ctx.lineWidth = 3;
                ctx.moveTo(drawPts[0].x, drawPts[0].y);
                for(let p of drawPts) ctx.lineTo(p.x, p.y);
                ctx.stroke();

                // Active dot
                let last = drawPts[drawPts.length-1];
                ctx.beginPath();
                ctx.arc(last.x, last.y, 6, 0, Math.PI*2);
                ctx.fillStyle = '#FFF';
                ctx.shadowColor = '#00FFAA';
                ctx.shadowBlur = 10;
                ctx.fill();
                ctx.shadowBlur = 0; // reset
            }}

            if (this.progress < 1.0) {{
                this.rafId = requestAnimationFrame(() => this.draw(ctx, w, h));
            }}
        }}
    }},
    2: {{
        simulation: null,
        start(ctx, w, h) {{
            ctx.clearRect(0,0,w,h);
            document.getElementById('main-canvas').style.display = 'none';
            const container = document.getElementById('d3-container');
            container.style.display = 'block';
            container.innerHTML = '';
            
            // Normalize radius sizes visually
            const maxVal = Math.max(...BETWEENNESS_DATA.map(d => d.val)) || 0.05;
            
            const nodes = BETWEENNESS_DATA.map((d, i) => ({{ 
                id: d.id, 
                obj: d, 
                isCenter: i === 0, 
                r: i===0 ? 30 : Math.max(10, (d.val / maxVal) * 20 + 8) 
            }}));
            
            const links = nodes.slice(1).map(n => ({{ source: nodes[0].id, target: n.id, value: 1 }}));
            // Add some cross links for network feel
            if (nodes.length > 5) {{
                links.push({{ source: nodes[1].id, target: nodes[2].id }});
                links.push({{ source: nodes[3].id, target: nodes[4].id }});
                links.push({{ source: nodes[2].id, target: nodes[5].id }});
            }}

            const svg = d3.select('#d3-container').append('svg')
                .attr('width', w).attr('height', h);
            
            const tooltip = d3.select("#d3-tooltip");

            this.simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(links).id(d => d.id).distance(100))
                .force("charge", d3.forceManyBody().strength(-300))
                .force("center", d3.forceCenter(w / 2, h / 2));

            const link = svg.append("g").attr("stroke", "#555").attr("stroke-opacity", 0.6)
                .selectAll("line").data(links).join("line")
                .attr("stroke-width", d => Math.sqrt(d.value));

             const node = svg.append("g").attr("stroke", "#0e1117").attr("stroke-width", 1.5)
                .selectAll("circle").data(nodes).join("circle")
                .attr("r", d => d.r)
                .attr("fill", d => d.isCenter ? "#00FFAA" : "#5591c7")
                .on("mouseover", function(event, d) {{
                    d3.select(this).attr("stroke", "#FFF").attr("stroke-width", 3);
                    tooltip.style("display", "block")
                        .html(`ID: <b>${{d.id}}</b><br>Betweenness: ${{d.obj.percentile}} %ile`)
                        .style("left", (event.pageX - document.getElementById('d3-container').getBoundingClientRect().left + 15) + "px")
                        .style("top", (event.pageY - document.getElementById('d3-container').getBoundingClientRect().top - 15) + "px");
                }})
                .on("mouseout", function() {{
                    d3.select(this).attr("stroke", "#0e1117").attr("stroke-width", 1.5);
                    tooltip.style("display", "none");
                }});

            const self = this;
            svg.selectAll("circle").call(d3.drag()
                .on("start", function(event, d) {{
                    if (!event.active) self.simulation.alphaTarget(0.3).restart();
                    d.fx = d.x; d.fy = d.y;
                }})
                .on("drag", function(event, d) {{ d.fx = event.x; d.fy = event.y; }})
                .on("end", function(event, d) {{
                    if (!event.active) self.simulation.alphaTarget(0);
                    d.fx = null; d.fy = null;
                }}));

            this.simulation.on("tick", () => {{
                link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
                node.attr("cx", d => d.x).attr("cy", d => d.y);
            }});
        }},
        stop() {{
            if(this.simulation) this.simulation.stop();
            document.getElementById('d3-container').innerHTML = '';
            document.getElementById('d3-container').style.display = 'none';
            document.getElementById('d3-tooltip').style.display = 'none';
            document.getElementById('main-canvas').style.display = 'block';
        }}
    }},
    0: {{
        rafId: null,
        progress: 0,
        start(ctx, w, h) {{
            this.progress = 0;
            this.draw(ctx, w, h);
        }},
        stop() {{ if (this.rafId) cancelAnimationFrame(this.rafId); }},
        draw(ctx, w, h) {{
            this.progress += 1;
            ctx.clearRect(0,0,w,h);
            ctx.font = "14px Courier New";
            ctx.fillStyle = "rgba(0, 255, 170, 0.7)";
            const streams = ["BMY US", "CELG US", "TWX US", "T US", "MSFT US", "ATVI US", "DIS US", "FOXA US"];
            for(let i=0; i<8; i++) {{
                let y = (this.progress * 1.5 + i*40) % h;
                let x = w * 0.2 + (i%3)*w*0.3;
                let alpha = Math.sin(this.progress * 0.05 + i) * 0.5 + 0.5;
                ctx.fillStyle = `rgba(0, 255, 170, ${{alpha}})`;
                ctx.fillText(`[INGEST] ${{streams[i]}} -> Node_${{i*128}}`, x, y);
            }}
            this.rafId = requestAnimationFrame(() => this.draw(ctx, w, h));
        }}
    }},
    3: {{
        rafId: null, progress: 0,
        start(ctx, w, h) {{ this.progress = 0; this.draw(ctx, w, h); }},
        stop() {{ if (this.rafId) cancelAnimationFrame(this.rafId); }},
        draw(ctx, w, h) {{
            this.progress += 0.05;
            ctx.clearRect(0,0,w,h);
            const cols = 12; const rows = 8;
            const size = Math.min(w/cols, h/rows) * 0.8;
            const offsetX = (w - cols*size)/2; const offsetY = (h - rows*size)/2;
            
            for(let r=0; r<rows; r++) {{
                for(let c=0; c<cols; c++) {{
                    let intensity = Math.sin(this.progress + r*0.5 + c*0.8) * 0.5 + 0.5;
                    // FinBERT heat (dark to orange/yellow)
                    let rColor = Math.floor(intensity * 255);
                    let gColor = Math.floor(intensity * 170);
                    ctx.fillStyle = `rgb(${{rColor}}, ${{gColor}}, 40)`;
                    ctx.fillRect(offsetX + c*size + 2, offsetY + r*size + 2, size-4, size-4);
                }}
            }}
            ctx.fillStyle = "#FFF"; ctx.font = "16px Courier New";
            ctx.fillText("Self-Attention Heatmap (Q x K^T)", offsetX, offsetY - 10);
            this.rafId = requestAnimationFrame(() => this.draw(ctx, w, h));
        }}
    }},
    4: {{
        rafId: null, progress: 0, particles: null,
        start(ctx, w, h) {{
            this.progress = 0;
            this.particles = Array.from({{length: 60}}).map((_, i) => ({{
                x: i%3===0 ? 0 : (i%3===1 ? w : w/2),
                y: i%3===0 ? h/2 : (i%3===1 ? h/2 : (i%2===0?0:h)),
                tx: w/2, ty: h/2,
                color: i%3===0 ? "#5591c7" : (i%3===1 ? "#fdab43" : "#00FFAA"),
                speed: Math.random()*0.02 + 0.01
            }}));
            this.draw(ctx, w, h);
        }},
        stop() {{ if (this.rafId) cancelAnimationFrame(this.rafId); }},
        draw(ctx, w, h) {{
            ctx.clearRect(0,0,w,h);
            ctx.fillStyle = "rgba(255,255,255,0.1)";
            ctx.beginPath(); ctx.arc(w/2, h/2, 40, 0, Math.PI*2); ctx.fill();
            
            this.particles.forEach(p => {{
                p.x += (p.tx - p.x) * p.speed;
                p.y += (p.ty - p.y) * p.speed;
                ctx.fillStyle = p.color;
                ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, Math.PI*2); ctx.fill();
                
                // Reposition when they arrive
                if(Math.abs(p.x - p.tx) < 5 && Math.abs(p.y - p.ty) < 5) {{
                    p.x = p.tx + (Math.random()-0.5)*80;
                    p.y = p.ty + (Math.random()-0.5)*80;
                }}
            }});
            ctx.fillStyle = "#FFF"; ctx.fillText("Financial", w*0.1, h/2);
            ctx.fillText("Semantic", w*0.8, h/2);
            ctx.fillText("Network", w/2 - 25, h*0.2);
            this.rafId = requestAnimationFrame(() => this.draw(ctx, w, h));
        }}
    }},
    5: {{
        rafId: null, progress: 0,
        start(ctx, w, h) {{ this.progress = 0; this.draw(ctx, w, h); }},
        stop() {{ if (this.rafId) cancelAnimationFrame(this.rafId); }},
        draw(ctx, w, h) {{
            this.progress += 0.02;
            ctx.clearRect(0,0,w,h);
            
            function drawTree(x, y, level, angle, p) {{
                if(level === 0 || p < 0) return;
                const len = 40 * level;
                const nx = x + Math.cos(angle) * len;
                const ny = y + Math.sin(angle) * len;
                
                ctx.strokeStyle = `rgba(0, 255, 170, ${{Math.min(1, p)}})`;
                ctx.lineWidth = level;
                ctx.beginPath(); ctx.moveTo(x,y); ctx.lineTo(nx,ny); ctx.stroke();
                
                ctx.fillStyle = "#FFF";
                ctx.beginPath(); ctx.arc(nx, ny, 3, 0, Math.PI*2); ctx.fill();
                
                drawTree(nx, ny, level-1, angle - 0.5, p - 0.3);
                drawTree(nx, ny, level-1, angle + 0.5, p - 0.3);
            }}
            
            // Draw 3 ensemble trees
            drawTree(w*0.25, h*0.8, 4, -Math.PI/2, this.progress);
            drawTree(w*0.5, h*0.8, 4, -Math.PI/2, this.progress - 0.5);
            drawTree(w*0.75, h*0.8, 4, -Math.PI/2, this.progress - 1.0);
            
            if (this.progress > 2.5) this.progress = 0; // loop
            
            this.rafId = requestAnimationFrame(() => this.draw(ctx, w, h));
        }}
    }}
}};

// --- Stage Manager ---
const StageManager = {{
    currentStage: -1,
    
    init() {{
        const stepper = document.getElementById('stepper-list');
        for (let i=0; i<STAGES.length; i++) {{
            let div = document.createElement('div');
            div.className = 'step';
            div.id = 'step-' + i;
            div.innerText = STAGES[i].title;
            // No IntersectionObserver, directly driven by user click callbacks
            div.onclick = () => this.activateStage(i);
            stepper.appendChild(div);
        }}
        this.activateStage(0);
    }},
    
    activateStage(idx) {{
        if (this.currentStage !== -1 && STAGE_ANIMATIONS[this.currentStage]) {{
            STAGE_ANIMATIONS[this.currentStage].stop();
        }}
        this.currentStage = idx;
        
        // Ensure initial sizing
        setTimeout(() => triggerHeightResize(), 10);
        
        // Stepper Visuals Height Matching
        const step0 = document.getElementById('step-0');
        const line = document.getElementById('stepper-line');
        const fill = document.getElementById('stepper-fill');
        
        let startY = step0.offsetTop + 25;
        let activeEl = document.getElementById('step-' + idx);
        let endY = document.getElementById('step-' + (STAGES.length - 1)).offsetTop + 25;
        let currentY = activeEl.offsetTop + 25;
        
        line.style.top = startY + 'px';
        line.style.height = (endY - startY) + 'px';
        
        fill.style.top = startY + 'px';
        fill.style.height = (currentY - startY) + 'px';
        
        const stepCount = STAGES.length;
        for (let i=0; i<stepCount; i++) {{
            let cl = document.getElementById('step-' + i).classList;
            if (i < idx) {{ cl.add('completed'); cl.remove('active'); }}
            else if (i === idx) {{ cl.add('active'); cl.remove('completed'); }}
            else {{ cl.remove('active'); cl.remove('completed'); }}
        }}
        
        // Concept and Formula updates
        document.getElementById('concept-container').innerHTML = 
            "<div class='concept-title'>" + STAGES[idx].title + "</div>" + STAGES[idx].desc;
            
        let fText = STAGES[idx].formula;
        let fDiv = document.getElementById('formula-container');
        if (fText.startsWith('Coverage')) {{
            fDiv.innerHTML = fText;
        }} else {{
            try {{
                katex.render(fText, fDiv, {{ displayMode: true }});
            }} catch(e) {{
                fDiv.innerHTML = fText;
                setTimeout(() => {{
                    try {{ katex.render(fText, fDiv, {{ displayMode: true }}); }} catch(err){{}}
                }}, 100);
            }}
        }}
        
        // Canvas Setup
        const canvas = document.getElementById('main-canvas');
        const container = canvas.parentElement;
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
        const ctx = canvas.getContext('2d');
        
        // Start explicit animation
        STAGE_ANIMATIONS[idx].start(ctx, canvas.width, canvas.height);
        
        setTimeout(() => triggerHeightResize(), 100);
    }}
}};

document.addEventListener("DOMContentLoaded", () => {{
    StageManager.init();
    setTimeout(triggerHeightResize, 200);
}});
window.addEventListener('resize', triggerHeightResize);

</script>
</body>
</html>
"""
    return html_template.format(
        betweenness_json=json.dumps(betweenness_top10)
    )

st.write("### Internal Engine & Pipeline Architecture")

betweenness_top10 = load_betweenness_data()

html_str = build_pipeline_html(betweenness_top10)
st.components.v1.html(html_str, height=750, scrolling=False)
