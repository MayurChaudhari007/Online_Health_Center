const sendBtn = document.getElementById("send-btn");
const input = document.getElementById("user-input");
const chatBox = document.getElementById("chat-box");

function appendMessage(type, text) {
  const div = document.createElement("div");
  div.className = `message ${type === "user" ? "user-message" : "bot-message"}`;
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function sendMessage() {
  const message = input.value.trim();
  if (!message) return;
  appendMessage("user", message);
  input.value = "";

  fetch("/get_response", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  })
    .then((res) => res.json())
    .then((data) => appendMessage("bot", data.reply))
    .catch(() => appendMessage("bot", "⚠️ Error connecting to server."));
}

sendBtn.addEventListener("click", sendMessage);
input.addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendMessage();
});

//      auto scroll after clear chat ///////////

// document.addEventListener("DOMContentLoaded", function () {
//   // Check if URL contains ?cleared=1
//   if (window.location.search.includes("cleared=1")) {
//     const toastEl = document.getElementById("chatToast");
//     if (toastEl) {
//       const toast = new bootstrap.Toast(toastEl);
//       toast.show();
//     }

//     // Scroll down to chat section
//     const chatSection = document.getElementById("chat-section");
//     if (chatSection) {
//       chatSection.scrollIntoView({ behavior: "smooth" });
//     }
//   }
// });

document.addEventListener("DOMContentLoaded", function () {
  const storage = document.getElementById("storage_container");

  if (storage) {
    const toastEl = document.getElementById("chatToast");
    const toastBody = document.getElementById("toastMessage");
    if (toastEl && toastBody) {
      toastBody.innerText = "Chat is cleared!";
      const toast = new bootstrap.Toast(toastEl, { delay: 3000 }); // auto-hide in 3s
      toast.show();
    }

    let scrolled = false;

    try {
      storage.scrollIntoView({ behavior: "smooth" });
      scrolled = true;
    } catch (e) {
      scrolled = false;
    }
  }
});
