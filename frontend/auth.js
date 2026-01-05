// ==============================
// Save token after login
// ==============================
function saveToken(token) {
    localStorage.setItem("token", token);
}

// ==============================
// Check Login (protect pages)
// ==============================
function checkLogin() {
    const token = localStorage.getItem("token");
    if (!token) {
        window.location.href = "login.html";
    }
}

// ==============================
// Logout
// ==============================
function logout() {
    localStorage.removeItem("token");
    window.location.href = "login.html";
}

// ==============================
// Helper to call protected APIs
// ==============================
async function apiRequest(url, method = "GET", body = null) {
    const token = localStorage.getItem("token");

    const options = {
        method: method,
        headers: {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + token
        }
    };

    if (body) {
        options.body = JSON.stringify(body);
    }

    const res = await fetch(url, options);

    // Token expired or missing → logout user
    if (res.status === 401) {
        localStorage.removeItem("token");
        window.location.href = "login.html";
    }

    return res.json();
}
