let dashboardData, trendChart, topChart, rocChart, forecastChart;
const fmt = n => Number(n).toLocaleString("en-PH", {maximumFractionDigits:0});
const pct = n => (n*100).toFixed(1)+"%";
async function init(){
  const res=await fetch("/api/dashboard"); dashboardData=await res.json();
  document.getElementById("cityCount").textContent=fmt(dashboardData.total_cities);
  document.getElementById("totalPop").textContent=fmt(dashboardData.total_population);
  document.getElementById("avgPop").textContent=fmt(dashboardData.average_population);
  renderCharts(); renderModels(); renderRows(dashboardData.records); renderForecastOptions();
}
function renderCharts(){
  if(trendChart)trendChart.destroy();if(topChart)topChart.destroy();
  trendChart=new Chart(document.getElementById("trendChart"),{type:"line",data:{labels:dashboardData.years,datasets:[{label:"Total population",data:dashboardData.population_totals,borderColor:"#3975ef",backgroundColor:"#3975ef20",fill:true,tension:.35,pointRadius:5}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{ticks:{callback:v=>fmt(v)}}}}});
  const top=dashboardData.top_cities.slice().reverse();
  topChart=new Chart(document.getElementById("topChart"),{type:"bar",data:{labels:top.map(x=>"City "+x.city_id),datasets:[{data:top.map(x=>x.pop_2024),backgroundColor:"#6f91e8",borderRadius:5}]},options:{indexAxis:"y",responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{ticks:{callback:v=>fmt(v)}}}}});
}
function renderModels(){
 const entries=Object.entries(dashboardData.models);
 document.getElementById("modelRows").innerHTML=entries.map(([name,m])=>`<tr><td class="metric">${name}</td><td>${pct(m.accuracy)}</td><td>${pct(m.precision)}</td><td>${pct(m.recall)}</td><td>${pct(m.f1)}</td><td>${pct(m.auc)}</td></tr>`).join("");
 const select=document.getElementById("matrixSelect");
 select.innerHTML=entries.map(([name])=>`<option>${name}</option>`).join("");
 select.onchange=showMatrix; showMatrix();
 if(rocChart)rocChart.destroy();
 const colors={"Categorical Naive Bayes":"#8b5cf6","Decision Tree":"#0f9d77","Random Forest":"#f08c28","Logistic Regression":"#e34b67"};
 const datasets=entries.map(([name,m])=>({label:name+" (AUC "+m.auc.toFixed(2)+")",data:m.roc.fpr.map((x,j)=>({x,y:m.roc.tpr[j]})),showLine:true,borderColor:colors[name]||"#3975ef",backgroundColor:colors[name]||"#3975ef",pointRadius:0,pointHoverRadius:4,borderWidth:3,tension:0,fill:false}));
 rocChart=new Chart(document.getElementById("rocChart"),{type:"scatter",data:{datasets:[...datasets,{label:"Random baseline",data:[{x:0,y:0},{x:1,y:1}],showLine:true,borderDash:[6,5],borderColor:"#9aa6b5",pointRadius:0,borderWidth:1.5}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:true,position:"bottom",labels:{usePointStyle:true,boxWidth:8,padding:14,font:{size:10}}}},scales:{x:{type:"linear",min:0,max:1,title:{display:true,text:"False Positive Rate"}},y:{min:0,max:1,title:{display:true,text:"True Positive Rate"}}}}});
 if(dashboardData.regression){document.getElementById("rmse").textContent=fmt(dashboardData.regression.rmse);document.getElementById("r2").textContent=(dashboardData.regression.r2*100).toFixed(1)+"%";}
}
function showMatrix(){
 const m=dashboardData.models[document.getElementById("matrixSelect").value];if(!m)return;
 const a=m.confusion_matrix;
 document.getElementById("matrix").innerHTML=`<div class="matrix"><div class="head"></div><div class="head">Predicted 0</div><div class="head">Predicted 1</div><div class="head">Actual 0</div><div>${a[0][0]}</div><div>${a[0][1]}</div><div class="head">Actual 1</div><div>${a[1][0]}</div><div>${a[1][1]}</div></div>`;
}
function renderRows(records){
 const body=document.getElementById("dataRows");
 body.innerHTML=records.slice(0,50).map(r=>`<tr><td>${r.city_id}</td><td>${fmt(r.pop_2010)}</td><td>${fmt(r.pop_2015)}</td><td>${fmt(r.pop_2020)}</td><td><b>${fmt(r.pop_2024)}</b></td><td>${Number(r.land_area_km2).toLocaleString()}</td><td>${fmt(r.dens_2020)}</td></tr>`).join("");
 document.getElementById("rowCount").textContent=`Showing ${Math.min(records.length,50)} of ${records.length} matching records.`;
}
document.getElementById("searchInput").addEventListener("input",e=>{
 const q=e.target.value.toLowerCase();
 const rows=dashboardData.records.filter(r=>Object.values(r).some(v=>String(v??"").toLowerCase().includes(q)));
 renderRows(rows);
});
function renderForecastOptions(){
 const select=document.getElementById("forecastSelect");
 select.innerHTML=dashboardData.forecast.map((r,i)=>`<option value="${i}">City ID ${r.city_id}</option>`).join("");
 select.onchange=drawForecast;drawForecast();
}
function drawForecast(){
 const row=dashboardData.forecast[Number(document.getElementById("forecastSelect").value)];
 const years=["2024","2025","2026","2027","2028","2029","2030"];
 const values=[row.actual_2024,...years.slice(1).map(y=>row[y])];
 if(forecastChart)forecastChart.destroy();
 forecastChart=new Chart(document.getElementById("forecastChart"),{type:"line",data:{labels:years,datasets:[{label:"Population",data:values,borderColor:"#16a47a",backgroundColor:"#16a47a18",fill:true,tension:.3}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{ticks:{callback:v=>fmt(v)}}}}});
}
document.getElementById("predictForm").addEventListener("submit",async e=>{
 e.preventDefault();const output=document.getElementById("predictionResults");output.textContent="Calculating predictions…";
 const payload={pop_2020:document.getElementById("p2020").value,land_area_km2:document.getElementById("area").value,dens_2020:document.getElementById("density").value};
 const response=await fetch("/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
 const data=await response.json();
 if(!response.ok){output.textContent=data.error||"Unable to calculate prediction.";return;}
 output.innerHTML='<div class="prediction-grid">'+Object.entries(data.predictions).map(([name,p])=>`<div class="prediction-card"><b>${name}</b><span>${p.prediction}</span><div class="footnote">Model confidence: ${pct(p.confidence)}</div></div>`).join("")+'</div>';
});
document.getElementById("uploadForm").addEventListener("submit",async e=>{
 e.preventDefault();
 const input=document.getElementById("csvFile"),button=document.getElementById("uploadButton"),status=document.getElementById("uploadStatus");
 if(!input.files.length){status.textContent="Choose a CSV file first.";status.className="upload-status error";return;}
 const formData=new FormData();formData.append("file",input.files[0]);
 button.disabled=true;button.textContent="Analyzing…";status.textContent="Checking columns and recalculating the dashboard and models…";status.className="upload-status";
 try{
  const response=await fetch("/api/upload",{method:"POST",body:formData});const data=await response.json();
  if(!response.ok)throw new Error(data.error||"The CSV could not be processed.");
  dashboardData=data;
  document.getElementById("cityCount").textContent=fmt(data.total_cities);document.getElementById("totalPop").textContent=fmt(data.total_population);document.getElementById("avgPop").textContent=fmt(data.average_population);
  renderCharts();renderModels();renderRows(data.records);renderForecastOptions();
  status.textContent=data.message+" Dashboard and algorithm results have been updated.";status.className="upload-status success";
 }catch(error){status.textContent=error.message;status.className="upload-status error";}
 finally{button.disabled=false;button.textContent="Upload and analyze";}
});
init().catch(err=>{console.error(err);document.querySelector(".main").insertAdjacentHTML("afterbegin",'<div class="notice">Unable to load dashboard data. Check the Flask terminal for details.</div>')});
