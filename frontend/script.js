async function predict() {
    const resultBox = document.getElementById("result");
    resultBox.innerHTML = "Loading...";

    try {
        const fieldIds = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'];
        const data = {};

        for (const id of fieldIds) {
            const val = document.getElementById(id).value.trim();
            if (!val) {
                resultBox.innerHTML = `⚠️ Field "${id}" is empty.`;
                return;
            }
            data[id] = id === 'oldpeak' ? parseFloat(val) : parseInt(val, 10);
            if (isNaN(data[id])) {
                resultBox.innerHTML = `❌ Invalid value for "${id}": ${val}`;
                return;
            }
            if (id === 'slope') {
                if (!Number.isInteger(data[id]) || data[id] < 0 || data[id] > 3) {
                    resultBox.innerHTML = '❌ Slope must be an integer 0, 1, 2, or 3.';
                    return;
                }
            }
            if (id === 'ca') {
                if (!Number.isInteger(data[id]) || data[id] < 0 || data[id] > 4) {
                    resultBox.innerHTML = '❌ CA must be an integer between 0 and 4.';
                    return;
                }
            }
            if (id === 'cp') {
                if (!Number.isInteger(data[id]) || data[id] < 0 || data[id] > 3) {
                    resultBox.innerHTML = '❌ Chest Pain Type must be an integer 0, 1, 2, or 3.';
                    return;
                }
            }
        }

        console.log("Sending:", JSON.stringify(data, null, 2));

        const res = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
            mode: "cors"
        });

        console.log("Response:", res.status);

        if (!res.ok) {
            const errorText = await res.text();
            throw new Error(`HTTP ${res.status}: ${errorText}`);
        }

        const out = await res.json();
        console.log("Result:", out);
        console.log("SHAP Values:", out.shap_values);
        console.log("Explanation:", out.explanation);

        const probability = out.probability * 100;
        const riskLevel = out.risk_level; // Use API risk_level instead of hardcoded logic
        const riskColor = riskLevel === "high" ? "#e63946" : riskLevel === "moderate" ? "#f77f00" : "#2a9d8f";
        const riskEmoji = riskLevel === "high" ? "🔴" : riskLevel === "moderate" ? "🟡" : "🟢";
        const riskText = riskLevel === "high" ? "HIGH RISK" : riskLevel === "moderate" ? "MODERATE RISK" : "LOW RISK";

        let resultHtml =
            `<div style="padding: 15px; border-radius: 8px; background: ${riskColor}22; border: 2px solid ${riskColor};">` +
            `<b style="font-size: 18px; color: ${riskColor};">${riskEmoji} ${riskText}</b><br><br>` +
            `<b>Prediction:</b> ${out.prediction === 1 ? "❤️ Heart Disease Detected" : "💚 No Heart Disease"}<br>` +
            `<b>Probability:</b> ${probability.toFixed(2)}%<br>`;
        
        resultHtml += `</div>`;
        
        // Add SHAP feature importance section
        console.log("Checking SHAP values...", out.shap_values);
        console.log("Has top_features?", out.shap_values?.top_features);
        console.log("Top features length:", out.shap_values?.top_features?.length);
        
        if (out.shap_values && out.shap_values.top_features && out.shap_values.top_features.length > 0) {
            console.log("✅ Rendering SHAP section with", out.shap_values.top_features.length, "features");
            resultHtml += `<div style="margin-top: 20px; padding: 15px; background: #1a1a1a; border-radius: 8px; border: 1px solid #444;">`;
            resultHtml += `<h3 style="margin-top: 0; color: #fff;">🔍 Feature Importance (SHAP Analysis)</h3>`;
            
            out.shap_values.top_features.forEach((feature, idx) => {
                const isPositive = feature.shap_contribution > 0;
                const color = isPositive ? '#e63946' : '#2a9d8f';
                const arrow = isPositive ? '⬆️' : '⬇️';
                
                resultHtml += `<div style="margin: 10px 0; padding: 10px; background: #222; border-left: 4px solid ${color}; border-radius: 4px; color: #fff;">`;
                resultHtml += `<b>${arrow} #${feature.rank}: ${feature.feature}</b><br>`;
                resultHtml += `<span style="font-size: 13px; color: #ddd;">Value: ${feature.value} | Impact: ${feature.impact}</span><br>`;
                resultHtml += `<span style="font-size: 12px; color: #aaa;">SHAP contribution: ${feature.shap_contribution > 0 ? '+' : ''}${(feature.shap_contribution * 100).toFixed(2)}%</span>`;
                resultHtml += `</div>`;
            });
            
            resultHtml += `</div>`;
        } else {
            console.log("❌ SHAP section NOT rendered - missing or empty top_features");
        }
        
        resultBox.innerHTML = resultHtml;

        // Draw SHAP chart if available
        if (out.shap_values && out.shap_values.top_features && out.shap_values.top_features.length > 0) {
            console.log("📊 Drawing SHAP chart with", out.shap_values.top_features.length, "features");
            try {
                drawShapChart(out.shap_values.top_features);
                console.log("✅ SHAP chart drawn successfully");
            } catch (chartErr) {
                console.error("❌ Error drawing SHAP chart:", chartErr);
            }
        } else {
            console.log("⚠️ No SHAP data to draw chart");
            // Hide canvas if no SHAP data
            const canvas = document.getElementById("shapChart");
            if (canvas) canvas.style.display = 'none';
        }

    } catch (err) {
        console.error("Error:", err);
        resultBox.innerHTML = `❌ Request failed: ${err.message || err}`;
        // Hide canvas on error
        const canvas = document.getElementById("shapChart");
        if (canvas) canvas.style.display = 'none';
    }
}

let shapChart = null;

function drawShapChart(topFeatures) {
    const canvas = document.getElementById("shapChart");
    if (!canvas) {
        console.error("❌ Canvas element #shapChart not found");
        return;
    }
    
    console.log("📊 Canvas found, getting context...");
    const ctx = canvas.getContext("2d");
    
    if (shapChart) {
        console.log("🗑️ Destroying previous chart");
        shapChart.destroy();
    }

    // Extract labels and values from top_features
    console.log("📊 Processing", topFeatures.length, "features");
    const labels = topFeatures.map(f => f.feature);
    const values = topFeatures.map(f => f.shap_contribution);
    console.log("📊 Labels:", labels);
    console.log("📊 Values:", values);
    
    console.log("📊 Creating Chart.js instance...");
    
    try {
        shapChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "SHAP Contribution",
                    data: values,
                    backgroundColor: values.map(v => v > 0 ? "#e63946" : "#2a9d8f"),
                    borderColor: values.map(v => v > 0 ? "#c62839" : "#228b7a"),
                    borderWidth: 2
                }]
            },
            options: {
                indexAxis: 'y', // Horizontal bar chart
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Feature Contributions to Prediction',
                        font: { size: 16, weight: 'bold' },
                        color: '#fff'
                    },
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const feature = topFeatures[context.dataIndex];
                                return [
                                    `Impact: ${feature.impact}`,
                                    `Value: ${feature.value}`,
                                    `Contribution: ${(feature.shap_contribution * 100).toFixed(2)}%`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'SHAP Value (Contribution to Prediction)',
                            color: '#fff'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#fff'
                        }
                    },
                    y: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#fff'
                        }
                    }
                }
            }
        });
        
        console.log("✅ Chart created successfully");
        
        // Make canvas visible
        canvas.style.display = 'block';
        canvas.style.marginTop = '20px';
        console.log("✅ Canvas made visible");
        
    } catch (err) {
        console.error("❌ Error creating Chart.js:", err);
        throw err;
    }
}