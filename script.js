// OTP Auto-focus
document.querySelectorAll(".otp-digit").forEach((input, index, inputs) => {
  input.addEventListener("input", (e) => {
    if (e.target.value && index < inputs.length - 1) {
      inputs[index + 1].focus();
    }
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Backspace" && !e.target.value && index > 0) {
      inputs[index - 1].focus();
    }
  });
});

function sendOTP() {
  const username = document.getElementById("username").value;
  const password = document.getElementById("password").value;

  if (!username || !password) {
    alert("Please enter username and password");
    return;
  }

  document.getElementById("loginForm").classList.add("hidden");
  document.getElementById("otpForm").classList.remove("hidden");
  alert("OTP sent to your device! (Demo: Use 123456)");
}

function verifyOTP() {
  const otpDigits = document.querySelectorAll(".otp-digit");
  const otp = Array.from(otpDigits)
    .map((input) => input.value)
    .join("");

  if (otp.length !== 6) {
    alert("Please enter complete OTP");
    return;
  }

  // Demo verification
  const userType = document.getElementById("userType").value;
  const username = document.getElementById("username").value;

  document.getElementById("loginPage").classList.remove("active");
  document.getElementById("mainApp").classList.add("active");
  document.getElementById("userName").textContent = `Welcome, ${username}`;

  // Show relevant sections based on user type
  if (userType === "patient") {
    document.getElementById("patientDashboard").classList.remove("hidden");
    document.getElementById("aiChatTab").style.display = "block";
    document.getElementById("epidemicTab").style.display = "block";
    document.getElementById("opdTab").style.display = "block";
    document.getElementById("recordsTab").style.display = "block";
  } else if (userType === "doctor") {
    document.getElementById("doctorDashboard").classList.remove("hidden");
    document.getElementById("trackingTab").style.display = "block";
    document.getElementById("scheduleTab").style.display = "block";
  } else if (userType === "admin") {
    document.getElementById("aiChatTab").style.display = "block";
    document.getElementById("epidemicTab").style.display = "block";
    document.getElementById("opdTab").style.display = "block";
    document.getElementById("recordsTab").style.display = "block";
    document.getElementById("doctorsTab").style.display = "block";
    document.getElementById("trackingTab").style.display = "block";
    document.getElementById("scheduleTab").style.display = "block";
  }
}

function backToLogin() {
  document.getElementById("loginForm").classList.remove("hidden");
  document.getElementById("otpForm").classList.add("hidden");
}

function logout() {
  document.getElementById("mainApp").classList.remove("active");
  document.getElementById("loginPage").classList.add("active");
  document.getElementById("loginForm").classList.remove("hidden");
  document.getElementById("otpForm").classList.add("hidden");
  document
    .querySelectorAll(".otp-digit")
    .forEach((input) => (input.value = ""));

  // Reset user-specific displays
  document.getElementById("patientDashboard").classList.add("hidden");
  document.getElementById("doctorDashboard").classList.add("hidden");
  document.getElementById("aiChatTab").style.display = "none";
  document.getElementById("epidemicTab").style.display = "none";
  document.getElementById("opdTab").style.display = "none";
  document.getElementById("recordsTab").style.display = "none";
  document.getElementById("trackingTab").style.display = "none";
  document.getElementById("scheduleTab").style.display = "none";
}

function showPage(pageName) {
  document.querySelectorAll(".page-content").forEach((page) => {
    page.classList.remove("active");
  });
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.remove("active");
  });

  document.getElementById(pageName).classList.add("active");
  event.target.classList.add("active");
}

function sendMessage() {
  const input = document.getElementById("chatInput");
  const message = input.value.trim();

  if (!message) return;

  const chatContainer = document.getElementById("chatContainer");

  // User message
  const userMsg = document.createElement("div");
  userMsg.className = "chat-message user";
  userMsg.textContent = message;
  chatContainer.appendChild(userMsg);

  input.value = "";
  chatContainer.scrollTop = chatContainer.scrollHeight;

  // Show loading indicator
  const loadingMsg = document.createElement("div");
  loadingMsg.className = "chat-message ai";
  loadingMsg.innerHTML = "🤔 Analyzing your symptoms...";
  loadingMsg.id = "loadingMessage";
  chatContainer.appendChild(loadingMsg);
  chatContainer.scrollTop = chatContainer.scrollHeight;

  // AI response via API
  $.ajax({
    url: "http://localhost:5000/api/health/chat",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      symptoms: message,
    }),
    success: function (response) {
      // Remove loading message
      const loading = document.getElementById("loadingMessage");
      if (loading) loading.remove();

      if (response.success) {
        // Create detailed response
        const aiMsg = document.createElement("div");
        aiMsg.className = "chat-message ai";

        let responseHTML = `<strong>${response.risk_icon} Risk Assessment: ${response.risk_level}</strong><br>`;
        responseHTML += `<strong>Risk Score:</strong> ${(
          response.risk_score * 100
        ).toFixed(1)}%<br><br>`;

        if (response.matched_symptoms.length > 0) {
          responseHTML += `<strong>Detected Symptoms:</strong> ${response.matched_symptoms.join(
            ", ",
          )}<br><br>`;
        }

        responseHTML += `<strong>Recommended Department:</strong> ${response.department_recommendation}<br><br>`;

        responseHTML += `<strong>Advice:</strong><br>`;
        response.advice.forEach((item) => {
          responseHTML += `• ${item}<br>`;
        });

        aiMsg.innerHTML = responseHTML;
        chatContainer.appendChild(aiMsg);
      } else {
        const errorMsg = document.createElement("div");
        errorMsg.className = "chat-message ai";
        errorMsg.textContent =
          "Sorry, I encountered an error. Please try again.";
        chatContainer.appendChild(errorMsg);
      }

      chatContainer.scrollTop = chatContainer.scrollHeight;
    },
    error: function (xhr, status, error) {
      // Remove loading message
      const loading = document.getElementById("loadingMessage");
      if (loading) loading.remove();

      const errorMsg = document.createElement("div");
      errorMsg.className = "chat-message ai";
      errorMsg.textContent =
        "⚠️ Unable to connect to the server. Please ensure the Flask API is running on http://localhost:5000";
      chatContainer.appendChild(errorMsg);

      chatContainer.scrollTop = chatContainer.scrollHeight;
      console.log("Error:", error);
    },
  });
}

function loadEpidemicPrediction() {
  // Show loading
  document.getElementById("epidemicLoading").classList.remove("hidden");
  document.getElementById("epidemicResults").classList.add("hidden");

  // Call API
}

function bookAppointment() {
  const department = document.getElementById("department").value;
  const doctor = document.getElementById("doctorSelect").value;
  const date = document.getElementById("appointmentDate").value;
  const time = document.getElementById("appointmentTime").value;

  if (!date) {
    alert("Please select a date");
    return;
  }

  alert(
    `Appointment booked successfully!\n\nDepartment: ${department}\nDoctor: ${doctor}\nDate: ${date}\nTime: ${time}\n\nYou will receive a confirmation SMS/Email shortly.`,
  );
}

// Set minimum date for appointment
document.addEventListener("DOMContentLoaded", function () {
  const today = new Date().toISOString().split("T")[0];
  const appointmentDate = document.getElementById("appointmentDate");
  if (appointmentDate) {
    appointmentDate.setAttribute("min", today);
  }

  // Chat enter key
  const chatInput = document.getElementById("chatInput");
  if (chatInput) {
    chatInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") sendMessage();
    });
  }

  // Initialize tabs visibility
  document.getElementById("aiChatTab").style.display = "none";
  document.getElementById("epidemicTab").style.display = "none";
  document.getElementById("opdTab").style.display = "none";
  document.getElementById("recordsTab").style.display = "none";
  document.getElementById("trackingTab").style.display = "none";
  document.getElementById("scheduleTab").style.display = "none";
});
