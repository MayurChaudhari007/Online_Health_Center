//  For auto scroll

document.addEventListener("DOMContentLoaded", function () {
  const report = document.getElementById("report");

  if (report) {
    let scrolled = false;

    try {
      report.scrollIntoView({ behavior: "smooth" });
      scrolled = true;
    } catch (e) {
      scrolled = false;
    }

    
    // setTimeout(() => {
    //   const nearReport = window.scrollY > report.offsetTop - window.innerHeight;
    //   if (!nearReport) {
    //     const toastEl = document.getElementById("report-toast");
    //     const toast = new bootstrap.Toast(toastEl);
    //     toast.show();
    //   }
    // }, 1000);
    const toastEl = document.getElementById("report-toast");
    const toast = new bootstrap.Toast(toastEl);
    toast.show();
  }
});




const patientNameElement = document.getElementById("pename");
let patientName = patientNameElement ? patientNameElement.textContent.trim() : "Patient";



document.getElementById("myForm").addEventListener("submit", function () {
  const btn = document.getElementById("predictBtn");
  const loading = document.getElementById("loading");
  
  btn.disabled = true;
  loading.style.display = "block";
});

//   / / / / / / / / / / / / / / / / // / / / / / / / / / / / / / / / / / / /  //  / /  / 


// document.getElementById("myForm").addEventListener("submit", async function (e) {
//   e.preventDefault();

//   const btn = document.getElementById("predictBtn");
//   const loading = document.getElementById("loading");
//   const errorBox = document.getElementById("errorBox");
//   const resultBox = document.getElementById("resultBox");

//   btn.disabled = true;
//   loading.style.display = "block";
//   errorBox.textContent = "";
//   resultBox.innerHTML = "";

//   try {
//     const formData = new FormData(this);
//     const response = await fetch("/ai_predict", {
//       method: "POST",
//       body: formData
//     });

//     const data = await response.json();

//     if (!response.ok) {
//       throw new Error(data.error || "Unknown error");
//     }

//     // ✅ Render result dynamically
//     resultBox.innerHTML = `
//       <div class="card p-3 shadow-sm">
//         <h4>🩺 AI Medical Report</h4>
//         <p><strong>Generated:</strong> ${data.report_time}</p>
//         <p><strong>Patient:</strong> ${data.result.patient_name || "N/A"}</p>
//         <p><strong>Predicted Disease:</strong> ${data.result.predicted_disease || "N/A"}</p>
//       </div>
//     `;

//   } catch (err) {
//     errorBox.textContent = err.message || "⚠️ Failed to get prediction.";
//   } finally {
//     btn.disabled = false;
//     loading.style.display = "none";
//   }
// });



// console.log(patientName);


document.addEventListener("DOMContentLoaded", function () {
  const pdfBtn = document.getElementById("downloadPDF");
  
  
  if (pdfBtn) {
    pdfBtn.addEventListener("click", function () {
      const reportElement = document.getElementById("report");

      if (!reportElement) {
        alert("Report not found!");
        return;
      }

      const opt = {
        margin: [0.5, 0.5, 0.5, 0.5], // top, left, bottom, right
        filename: `${patientName}_AI_Medical_Report.pdf`,
        image: { type: "jpeg", quality: 0.98 },
        html2canvas: { scale: 2, useCORS: true }, // useCORS ensures external CSS/images load
        jsPDF: { unit: "in", format: "a4", orientation: "portrait" },
        pagebreak: { mode: ["avoid-all", "css", "legacy"] }, // prevents cutting sections
      };

      // Use DOM element, not innerHTML
      html2pdf().set(opt).from(reportElement).save();
    });
  }
});
